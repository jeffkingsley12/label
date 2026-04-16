"""
Luganda FastText Training Data Generator  ─  v2.1
==================================================
Fixes over v2.0:
  • [CRITICAL] OOS quota > unique templates crash — early-return path via rng.sample()
  • [CRITICAL] amt_pair[1] was dead data — exposed as {amt_w_num} slot
  • [MEDIUM]   _compute_quotas() now returns dict[str,int] — eliminates positional coupling
  • [MEDIUM]   inject_noise() step order fixed: code-switch before lowercase (avoids mixed-case artefact)
  • [MEDIUM]   Slang replacement uses single combined alternation regex (O(n) not O(15n))
  • [MEDIUM]   generate_for_label() stall detection — fails fast instead of busy-looping
  • [MEDIUM]   Template pre-validation at import time (_validate_templates)
  • [MINOR]    All probability magic numbers moved to Config
  • [MINOR]    greet_r dead slot removed from fill_slots()
  • [MINOR]    assert replaced with explicit if/raise (safe under python -O)
  • [MINOR]    argparse wired to __main__ — no source edits needed for common params
  • [MINOR]    File write wrapped in try/except OSError
  • [MINOR]    print_quality_report() rewritten as a single pass over data
"""

import argparse
import math
import random
import re
from collections import Counter
from typing import NamedTuple

# ── Local linguistic module ────────────────────────────────────────────────────
try:
    from elision import LugandaElisionHandler
except ImportError as exc:
    raise ImportError(
        "elision.py not found. Place it in the same directory as _gen.py."
    ) from exc

# Instantiate once — compiled regex patterns inside are reused for every call.
_ELISION = LugandaElisionHandler()


# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  (all tuneable constants in one place)
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """Central home for every magic number in the pipeline."""
    # Morphological elision
    ELISION_PROB:      float = 0.35   # fraction of lines that get elision applied
    ELISION_SPLIT:     float = 0.50   # within elision: P(contract) vs P(expand)

    # Noise injection
    NOISE_PROB:        float = 0.20   # fraction of lines that receive noise
    LOWERCASE_PROB:    float = 0.30   # sub-chance: full lowercase
    PUNCT_DROP_PROB:   float = 0.30   # sub-chance: strip trailing punctuation
    CODE_SWITCH_PROB:  float = 0.20   # sub-chance: one Luganda verb → English

    # Generation engine
    MAX_ATTEMPT_MULT:  int   = 300    # max_attempts = quota * MAX_ATTEMPT_MULT
    STALL_LIMIT_MULT:  int   = 10     # stall break = min(500, quota * STALL_LIMIT_MULT)
    STALL_LIMIT_CAP:   int   = 500


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LEXICON
# ══════════════════════════════════════════════════════════════════════════════

class AmountPair(NamedTuple):
    """
    word:    Luganda word form  (used by {amt_w})
    numeric: digit string       (used by {amt_w_num} — paired literal for the word)
    """
    word:    str
    numeric: str


AMOUNTS: list[AmountPair] = [
    # Hundreds
    AmountPair("ebikumi bibiri",           "200"),
    AmountPair("ebikumi bisatu",           "300"),
    AmountPair("ebikumi bitaano",          "500"),
    AmountPair("ekikumi",                  "100"),
    # Thousands
    AmountPair("emitwalo ebiri",           "2000"),
    AmountPair("emitwalo esatu",           "3000"),
    AmountPair("emitwalo etaano",          "5000"),
    AmountPair("emitwalo kkumi",           "10000"),
    AmountPair("emitwalo kkumi n'etaano",  "15000"),
    AmountPair("emitwalo makumi abiri",    "20000"),
    AmountPair("emitwalo makumi asatu",    "30000"),
    AmountPair("emitwalo makumi ataano",   "50000"),
    # Millions
    AmountPair("omutwalo gumu",            "1000000"),
    AmountPair("emitwalo egiri",           "2000000"),
    AmountPair("emirundi esatu",           "3000000"),
]

NUMERIC_AMOUNTS = [
    "100", "200", "250", "500", "1000", "1500", "2000", "2500",
    "3000", "5000", "7500", "10000", "15000", "20000", "25000",
    "30000", "50000", "75000", "100000", "200000", "500000",
]

# Slang aliases used by inject_noise() — maps canonical numeric string → slang
NUMERIC_SLANG: dict[str, str] = {
    "1000":  "1k",   "2000":  "2k",   "3000":  "3k",
    "5000":  "5k",   "7500":  "7.5k", "10000": "10k",
    "15000": "15k",  "20000": "20k",  "25000": "25k",
    "30000": "30k",  "50000": "50k",  "75000": "75k",
    "100000":"100k", "200000":"200k", "500000":"500k",
}

PEOPLE = [
    "maama", "taata", "muko", "muganda wange", "omukozi", "ssebo",
    "nyabo", "mukwano wange", "omwami wange", "mwanaange",
    "mukama wange", "ow'omu maka", "munnaange", "omukwano",
    "muganda", "mwana wange", "mukyala wange", "ow'ekibinja",
]

PROVIDERS = ["Airtel", "MTN", "Stanbic", "DFCU", "Centenary", "Equity", "PostBank"]

BILL_TYPES = [
    "amayengo g'amazzi",       "amayengo g'amasanye",     "ssuukali",
    "ffiiri y'amasanye",       "ffiiri y'amazzi",          "ffiiri ya NWSC",
    "ffiiri ya Umeme",          "ffene ya ggwanga",         "amateekkwa g'essomero",
    "omusingo gw'ezzukuka",     "amateekkwa g'obusuubuzi",
    "ffiiri ya DStv",           "amateekkwa g'akavuulu",
]

ACCOUNTS = [
    "akawunti yange",        "simu yange",             "nomba yange",
    "akawunti ya savings",   "nomba y'akawunti",        "akawunti ya mobile money",
    "akawunti ya {provider}",
]

UI_ELEMENTS = [
    "menyu",             "skrini eyedda",     "lupapula",
    "fayiro",            "olukalala",          "pulogulaamu",
    "akabonero",         "essimu",             "ekitundu",
    "ekigendererwa",     "olusozi lw'ebintu",
    "skrini enkulu",     "ekifaananyi",        "entebbe",
]

GREETINGS_MORNING  = ["Wasuze otyanno", "Wasuze ennyo",       "Osiibye otyanno"]
GREETINGS_GENERAL  = ["Ki kati",         "Gyebale ko",          "Oli otyanno",
                       "Osibye otegeera", "Nkusanyukira ddala"]
GREETINGS_RESPONSE = ["Bulungi",         "Ndi bulungi",         "Kale bulungi",
                       "Siiba bulungi nnyo"]
POLITENESS         = ["ssebo", "nyabo", "munnange", "bambi", "omwana"]

ITEMS_INFO = [
    "emmere",    "amazzi",    "ennyumba",   "ssente",     "obuyambi",
    "omusawo",   "essomero",  "eggwanga",   "omuwendo",   "okusula",
    "obuguzi",   "akawunti",  "akabonero",  "amaanyi",    "ebintu",
    "emirimo",   "obulamu",   "eby'okwata",
]

PROBLEMS = [
    "okusindika ssente",     "okusasula ffiiri",      "okutuuka ku akawunti",
    "okufuna obuyambi",      "okubaako namba",          "okuzimba akawunti",
    "okuzimba simu",         "okulaba omuwendo",        "okuyingira mu pulogulaamu",
    "okugeza password",
]

POSITIVE_ADJ = [
    "bulungi nnyo", "kigenda mangu", "kikoze", "kiyamba nnyo",
    "kirina amaanyi", "kirondoola", "kituufu",
]

NEGATIVE_ADJ = [
    "tekikola bulungi", "kinafu nnyo", "kisaasaanya",
    "kifaayo", "tekimala", "kibonabona", "kikaabya",
]

Q_WORDS = ["Wa", "Lwaki", "Ani", "Kiki", "Engeri ki", "Edda di"]

# Code-switch substitution map used by inject_noise()
CODE_SWITCH_MAP: dict[str, str] = {
    "Sindiika": "Send",
    "Weereza":  "Send",
    "Sasula":   "Pay",
    "Njagala":  "I want",
    "Balansi":  "Balance",
    "Tuma":     "Send",
    "Fungula":  "Open",
    "Ddayo":    "Go back",
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TEMPLATE BANK
# ══════════════════════════════════════════════════════════════════════════════

TEMPLATES: dict[str, list[str]] = {

    # ── trx_transfer ──────────────────────────────────────────────────────────
    "__label__trx_transfer": [
        "Sindiika {amt_w} eri {person}.",
        "Sindiika {amt_n} eri {person}.",
        "Weereza {person} {amt_w}.",
        "Weereza {person} {amt_n}.",
        "Njagala okuweereza {person} {amt_w}.",
        "Njagala okuweereza {person} {amt_n}.",
        "Tuma {amt_w} eri {person} mangu.",
        "Tuma {amt_n} eri {person} mangu.",
        "Sindiika {person} {amt_w} ku {provider}.",
        "Sindiika {person} {amt_n} ku {provider}.",
        "Weereza {phone} {amt_n}.",
        "Sindiika {phone} {amt_w}.",
        "Njagala okutuma ssente eri {person}.",
        "Nsindiike {person} {amt_n} ku simu.",
        "Tuma {amt_n} eri {phone}.",
        "Weereza {person} {amt_w} ku nomba ye.",
        "Nkwagala oweereze {person} {amt_n}.",
        "Nkuwa {person} {amt_w}.",
        "Nkuwa {person} {amt_n} ku {provider}.",
        "Kola transfer ya {amt_n} eri {person}.",
        "Kola transfer ya {amt_w} eri {phone}.",
        "Sindiika ssente {amt_n} eri {person} mangu.",
        "{person} mpe {amt_w}.",
        "{person} mpe {amt_n} ku {provider}.",
        "Yambaako okuweereza {person} {amt_w}.",
        "Yambaako okuweereza {phone} {amt_n}.",
        "Nkusaba oweereze {person} {amt_n}.",
        "Nkusaba oweereze {phone} {amt_w}.",
        "Bwereza {person} {amt_n}.",
        "Sindiika {person} {amt_n} kati kati.",
        "Weereza {person} {amt_w} mangu nnyo.",
        "Nkwagala otume {amt_n} eri {person}.",
        "Fuba okusindiika {amt_w} eri {phone}.",
        "Sindiika {phone} {amt_n} ku {provider}.",
        "Njagala okusindiika {amt_n} eri {person} ku {provider}.",
    ],

    # ── trx_balance ───────────────────────────────────────────────────────────
    "__label__trx_balance": [
        "Ssente zange mmeka ezisigaddewo?",
        "Njagala okulaba balansi yange.",
        "Nkimanyi nti nnina {amt_w} ku {account}.",
        "Balansi yange eri mmeka?",
        "Okyusa {account} yange.",
        "Laba omuwendo gw'okusigalawo ku {account}.",
        "Mmeka gy'ennina ku {account}?",
        "Nkimanyi omuwendo ogusigaddewo.",
        "Okukyusa balansi ya {account}.",
        "Nkwagala ombuulire ssente eziri ku {account}.",
        "Mmeka ssente ezisigaddewo ku simu yange?",
        "Jukira {account} yange omuwendo.",
        "Nfuna okunyweza {account} yange.",
        "Mbulira mmeka ssente eziri ku {provider}.",
        "Laba mmeka eby'okusigalawo ku {account}.",
        "Ssente zange ziri wa?",
        "Gamba nti balansi yange eri mmeka.",
        "Nkusaba ondage omuwendo gw'okusigalawo.",
        "Kyusa {account} yange mangu.",
        "Kiki ekisigaddewo ku {account} yange?",
        "Nfuna laba balansi ya {provider}.",
        "Balansi ya {account} eri mmeka leero?",
        "Njagala okumanya ssente eziri ku {account}.",
        "Nkimanyi nti balansi yange eri {amt_n}.",
        "Funa omuwendo gw'okusigalawo ku {account}.",
        "Laba {account} yange olw'okumanya balansi.",
        "Njagala okwerabuka ku balansi ya {provider}.",
        "Mbulira nti nnina {amt_w} ku {account}?",
        "Kiki omuwendo ogusigaddewo ku {account}?",
        "Balaansi ya {account} etya leero?",
        "Nkwatira okukyusa balansi ya {account}.",
        "Nkuwe {account} yange omuwendo mangu.",
        "Wunyirira {account} yange.",
        "Funa omuwendo ku {account} ya {provider}.",
        "Nkusaba ondage balansi ya {account} ya {provider}.",
        "Laba omuwendo gusigaddewo ku {account} ya {provider}.",
        "Balansi ya {provider} eri mmeka ku {account}?",
        "Nkuwe okumanya ssente eziri ku {account}.",
        "Gamba nti nnina {amt_n} ku {account}?",
        "Kyusa {account} ya {provider} yange.",
        "Mbulira ssente ezisigaddewo ku {provider} wange.",
        "Ssente zange ku {provider} ziri mmeka?",
    ],

    # ── trx_payment ───────────────────────────────────────────────────────────
    "__label__trx_payment": [
        "Sasula {bill}.",
        "Njagala okusasula {bill}.",
        "Sasula {bill} mangu.",
        "Nkola okutuuka ku {bill}.",
        "Fuba okusasula {bill} eri {provider}.",
        "Sasula {bill} ya {amt_n}.",
        "Njagala okusasula {bill} wa {provider}.",
        "Kola okuliwa {bill}.",
        "Sasula ffiiri ya {provider}.",
        "Nkola okufiirwa {bill}.",
        "Mpe okusasula {bill}.",
        "Yambaako okuliwa {bill}.",
        "Sasula {bill} w'omu maka wange.",
        "Njagala okulipirira {bill}.",
        "Lipirira {bill} mangu.",
        "Sasula {bill} okusobola okutuukako.",
        "Kola payment ya {bill}.",
        "Fuba okusasula {bill} wa {amt_n}.",
        "Sasula {bill} ya {provider} eri {amt_n}.",
        "Nkuwa okusasula {bill}.",
        "Yambaako okusasula amateekkwa g'essomero.",
        "Sasula omusingo gw'ezzukuka mangu.",
        "Njagala okugaba ssente ez'amazzi.",
        "Kola okuliwa ffiiri ya Umeme.",
        "Sasula ffiiri ya NWSC eri {amt_n}.",
        "Nkusaba osasule {bill} eri {provider}.",
        "Kola okuliwa {bill} ya {amt_n}.",
        "Lipirira {bill} ya {provider} mangu.",
        "Sasula {bill} ku {provider} eri {amt_n}.",
        "Nkwagala osasule {bill}.",
    ],

    # ── soc_greeting ──────────────────────────────────────────────────────────
    "__label__soc_greeting": [
        "{greet_g}.",
        "{greet_g}, {polite}.",
        "{greet_g}, {polite}, otyanno?",
        "{greet_m}.",
        "{greet_m}, {polite}.",
        "Wasuze otyanno, {polite}?",
        "Osiibye otyanno, {polite}?",
        "Ki kati, {polite}?",
        "Oli otyanno {polite}?",
        "Osibye otegeera, {polite}?",
        "{greet_g} {polite}, nkusaba obuyambi.",
        "Gyebale, {polite}, nnaafayo.",
        "Wasuze ennyo, {polite}.",
        "Tukusanyukidde, {polite}.",
        "Nkusanyukira okukulaba, {polite}.",
        "Gwe oli, {polite}?",
        "Weeba, {polite}, nga otyanno?",
        "Ki kati {polite}, oli bulungi?",
        "Wasuze otyanno bonna.",
        "Nkusanyukira {polite}.",
        "{greet_m}, {polite}, osibye otyanno?",
        "Nkukusanyukira nnyo, {polite}.",
        "{greet_g}, {polite}, oba nga otyanno?",
        "Osibye otegeera, {polite}, nkusaba {item}.",
        "Oli wa {polite}?",
    ],

    # ── soc_gratitude ─────────────────────────────────────────────────────────
    "__label__soc_gratitude": [
        "Weebale nnyo.",
        "Weebale nnyo, {polite}.",
        "Gyebale nnyo.",
        "Nsanyuse nnyo, {polite}.",
        "Nsanyuse okuba naawe.",
        "Webale obuyambi bwo.",
        "Webale nnyo okumpa {item}.",
        "Weebale nnyo okunkola {item}.",
        "Nsanyuse nnyo n'obuyambi bwo.",
        "Katonda akuwe obuzibu, {polite}.",
        "Weebale nnyo, okunzimba.",
        "Nsanyuse nnyo okufuna obuyambi bwo.",
        "Webale {polite}, obuzibu bwakola bulungi.",
        "Nsanyuse nnyo nga kikoze.",
        "Weebale nnyo, kironda ddala.",
        "Gyebale ko nnyo {polite}.",
        "Nkusanyukira nnyo obuyambi bwo.",
        "Weebale okunkola nga bino bikoze.",
        "Nsanyuse nnyo n'okumbuulira.",
        "Weebale nnyo, ngenze.",
        "Weebale nnyo okunkuyamba ku {item}.",
        "Nsanyuse nnyo okufuna {item}.",
        "Katonda akuwe {polite}.",
        "Weebale nnyo, {polite}, okunkola.",
        "Nsanyuse nnyo n'ebikoze.",
        "Webale nnyo {person}.",
        "Weebale okuyamba ku {item}, {polite}.",
        "Nsanyuse nnyo n'okumpa obuyambi.",
        "Weebale {person}, kironda.",
        "Gyebale nnyo {person}.",
        "Nsanyuse nnyo okufunamu {item}.",
        "Webale nnyo okumpa {item}, {polite}.",
        "Nsanyuse nnyo naffe n'{item}.",
        "Weebale nnyo {person}, kigenze bulungi.",
        "Nsanyuse nnyo okufunamu obuyambi bwange.",
        "Weebale {polite} okunkola {item}.",
        "Nsanyuse nnyo, kikoze {person}.",
        "Webale nnyo okunkuyamba {polite}.",
        "Nsanyuse nnyo okukozesa {item}.",
        "Gyebale ko nnyo {person}, weebale.",
        "Nsanyuse nnyo ng'obuzibu bwakola, {polite}.",
        "Weebale {person} okunkuyamba ku {item}.",
        "Nsanyuse nnyo okukola n'{person}.",
        "Weebale {polite} n'obuyambi bwo ku {item}.",
        "Nsanyuse nnyo okufuna obuyambi bwo, {person}.",
        "Katonda akuwe {person} amazima.",
        "Gyebale ko {person}, nsanyuse nnyo.",
        "Weebale {person} ddala.",
        "Nsanyuse nnyo, {person}, kikoze.",
        "Weebale {polite} okunkola {item} mangu.",
        "Nsanyuse nnyo naffe n'okufuna {item}.",
    ],

    # ── nav_command ───────────────────────────────────────────────────────────
    "__label__nav_command": [
        "Ddayo emabega ku {ui}.",
        "Genda mu maaso ku {ui}.",
        "Fungula {ui} eno mangu.",
        "Genda ku {ui}.",
        "Nkwagala ogenda ku {ui}.",
        "Ddiramu {ui}.",
        "Nyiga ku {ui}.",
        "Salawo {ui}.",
        "Komyawo {ui}.",
        "Yimuka ku {ui}.",
        "Ddayo ku {ui} eyedda.",
        "Genda ddayo ku {ui}.",
        "Fungula {ui} luno.",
        "Nkwagala onfungulire {ui}.",
        "Nyiga ko ku {ui}.",
        "Genda ku ekitundu ky'{ui}.",
        "Nkwagala ogenda ku {ui} mangu.",
        "Tuuka ku {ui} eno.",
        "Bwino ku {ui}.",
        "Salawo {ui} eno.",
        "Ddayo emabega oluusi.",
        "Genda ku menyu enkulu.",
        "Fungula lukalala lw'ebintu.",
        "Genda ku skrini eyongedde.",
        "Nkusaba oddayo ku {ui}.",
        "Tuuka ku {ui} kati.",
        "Nyiga ko {ui} eno mangu.",
        "Nkwagala olabe {ui}.",
        "Funa {ui} mangu.",
        "Genda ku {ui} okufuna {item}.",
    ],

    # ── inf_question ──────────────────────────────────────────────────────────
    "__label__inf_question": [
        "{q} gye nnyinza okufuna {item}?",
        "Kiki ekikwata ku {item}?",
        "Ngamba {item} ki?",
        "Oyinza kunnyamba ku {item}?",
        "Nkusaba onnyambe ku {item}.",
        "Njagala okunanya ku {item}.",
        "Bwetyo otyanno ku {item}?",
        "{q} gye njyeko okutuuka ku {item}?",
        "Yambaako ku {item}.",
        "Nkusaba ondage {item}.",
        "Kiki ekivaako ku {item}?",
        "Engeri ki gy'okutuuka ku {item}?",
        "Oyambaako okutuukako ku {item}?",
        "Nkusaba ontegeeze ku {item}.",
        "Kiki ky'olina ku {item}?",
        "Mbulire ku {item}, {polite}.",
        "{q} gye nnyinza okufuna obuyambi ku {item}?",
        "Nga {item} bw'eba otyanno?",
        "Kiki ekitera okuba ne {item}?",
        "Nnyambe okutegeera {item}.",
        "Mmeka gy'okifunamu {item}?",
        "Oyinza okumbuulira ku {item}?",
        "Nkimanyi ntya ku {item}?",
        "{q} gy'okukolera ku {item}?",
        "Kiki ky'ekitaagibwa okutuuka ku {item}?",
        "Mbulira nti {item} bwekikoze otyanno.",
    ],

    # ── fdb_neg ───────────────────────────────────────────────────────────────
    "__label__fdb_neg": [
        "Tekikola bulungi.",
        "Kinafu nnyo.",
        "{problem} tekigenda.",
        "{problem} kinafu.",
        "Tekiyamba.",
        "Kitera okulemwa.",
        "Kino tekisiimibwa.",
        "Nga kino kibonerabonera.",
        "Sikoze, {polite}.",
        "Ekikoze bubi.",
        "Takola naye.",
        "Kino kikaabya nnyo.",
        "Sikyogerwa nayo.",
        "Kiggwaako ddala.",
        "Tekimala.",
        "Nga kibonabona.",
        "{problem} tekikoze bulungi.",
        "Nsanyuse mangu ku {problem}.",
        "Kibaako ekibi ku {problem}.",
        "Kino kizibu nnyo.",
        "Sikoze buli kiseera.",
        "Tekola naye.",
        "Nnyisa nnyo, kino kijjawo.",
        "Eky'obubi {problem} bwekibeera.",
        "Kino kijja njawulo.",
        "Sisobola kukozesa {item} bulungi.",
        "Nga {problem} kibonerabonera nnyo.",
        "{item} sikoze nga bw'eteekwa.",
        "Nkimu nnyo ku {problem}.",
        "Kibaako obuzibu ku {item}.",
        "Sikyogera nayo ku {item}.",
        "Kigwa oluusi buli kiseera.",
        "Tekitambula ku {item}.",
        "Wano walimu ekibi ku {problem}.",
        "Nga {item} kidda omusitaani.",
        "Tekikoleramu bulungi.",
        "Nzikiwa nnyo n'{item}.",
        "Kino kibonerabonera nnyo ku {problem}.",
        "{problem} kitera okugwa.",
        "Kinafu ddala ku {item}.",
        "Sikoze nga bw'eneeme {item}.",
        "Nga {problem} kikaabya nnyo.",
        "Nnyisa nnyo n'okukola {problem}.",
        "Kino tekikola nga bw'eteekwa.",
        "{item} sisobola kukozesebwa bulungi.",
    ],

    # ── fdb_pos ───────────────────────────────────────────────────────────────
    "__label__fdb_pos": [
        "Kikoze bulungi nnyo.",
        "Nsanyuse nnyo.",
        "Kigenda mangu nnyo.",
        "Kiyamba nnyo.",
        "Pulogulaamu eno yikoze {pos_adj}.",
        "Nsanyuse {pos_adj} n'ekikoze.",
        "Kigenda bulungi nnyo.",
        "Kino kituufu.",
        "Nsanyuse okukozesa {item}.",
        "Kino kirondoola nnyo.",
        "Kirina amaanyi nnyo.",
        "{problem} kikoze bulungi.",
        "Simu eno yikoze {pos_adj}.",
        "Pulogulaamu eno eyamba nnyo.",
        "Nkosezza obulungi, {polite}.",
        "Obudde bwakola bulungi.",
        "Amaanyi g'akayungirizo kitera okubeera {pos_adj}.",
        "Kituufu nnyo okukozesa.",
        "Kyendawo mangu nnyo.",
        "Kilabirira bulungi nnyo.",
        "Nsanyuse nnyo n'okukozesa pulogulaamu eno.",
        "Kikoze nga bw'ekyetaagibwa.",
        "Nsanyuse nnyo, {polite}.",
        "Kino kisanyusa nnyo.",
        "Kituufu bulungi.",
        "Kigenda bulungi nnyo ku {item}.",
        "{item} yakola bulungi nnyo.",
        "Nsanyuse nnyo n'eby'okukola {item}.",
        "Kirondoola nnyo ku {problem}.",
        "Kikoze {pos_adj} ku {item}.",
        "Nsanyuse nnyo okukozesa {item}.",
        "{item} yakola {pos_adj}.",
        "Kigenda mangu nnyo ku {item}.",
        "Pulogulaamu eno eyamba ku {item} {pos_adj}.",
        "Nsanyuse nnyo, {item} yakola.",
        "Kituufu ddala ku {item}.",
        "{problem} kikoze {pos_adj} leero.",
        "Nsanyuse nnyo n'obuzibu bwakola.",
        "Kirondoola nnyo ku {item}.",
        "Kirina amaanyi nnyo ku {problem}.",
        "Kigenda nga bw'eteekwa ku {item}.",
        "Nsanyuse nnyo okufuna {item}.",
        "Kikoze {pos_adj}, nsanyuse nnyo.",
        "{item} yakola {pos_adj} nnyo.",
    ],

    # ── oos (out-of-scope hard negatives) ─────────────────────────────────────
    # English / random / nonsense lines the model must reject at inference.
    # No slots used — these are verbatim so elision + noise are skipped.
    "__label__oos": [
        "Hello how are you doing today.",
        "The weather is nice today.",
        "Testing testing one two three.",
        "Random text with no meaning here.",
        "I like cats and dogs.",
        "What time is it now?",
        "This system is broken 12345.",
        "Abcde fghij klmno pqrst.",
        "Can you help me find the store?",
        "Please call me back later.",
        "I need directions to downtown.",
        "What is your name?",
        "Turn off the lights please.",
        "The meeting is at three pm.",
        "I forgot my password again.",
        "00000 00000 00000.",
        "Soccer is my favourite sport.",
        "Order pizza for delivery now.",
        "My flight is delayed by one hour.",
        "Remind me about tomorrow morning.",
        "Play some music for me.",
        "Set an alarm for seven am.",
        "Book a table for two people.",
        "Add milk to my shopping list.",
        "What movies are showing tonight.",
        "Tell me a joke please.",
        "Translate this text to English.",
        "How do I reset my phone?",
        "Search for cheap flights to Nairobi.",
        "What is the capital of France?",
        "Xzqw mno pqrst uvwxyz.",
        "1111 2222 3333 4444 5555.",
        "Is Kampala bigger than Entebbe?",
        "How many kilometers to Jinja?",
        "Open maps and find a restaurant.",
        "Create a new document for me.",
        "Read me the latest news headlines.",
        "What is five plus seven?",
        "How many days in February?",
        "Convert one dollar to Uganda shillings.",
        "What is the best restaurant in Kampala?",
        "How do I get to Entebbe airport?",
        "I need a taxi to the city centre.",
        "What time does the supermarket close?",
        "Can you recommend a good hotel?",
        "My computer is not working properly.",
        "I want to watch a movie tonight.",
        "What is the best route to Mbarara?",
        "Please help me with my homework.",
        "I need to buy groceries today.",
        "The bus leaves at six in the morning.",
        "Turn up the volume on the radio.",
        "I want to learn a new language.",
        "What is the population of Uganda?",
        "How do I apply for a passport?",
        "What are the visiting hours at the hospital?",
        "I need to renew my driving licence.",
        "Where can I buy fresh vegetables?",
        "Tell me about the history of Kampala.",
        "What is the temperature outside?",
        "Who won the football match yesterday?",   # extra lines so OOS quota ≥ 60
        "How do I charge my laptop faster?",
        "What channels are on DStv tonight?",
        "Can you set a reminder for me?",
    ],
}

# ── Label weights (must sum to 1.0) ────────────────────────────────────────────
LABEL_WEIGHTS: dict[str, float] = {
    "__label__trx_transfer":  0.17,
    "__label__trx_balance":   0.12,
    "__label__trx_payment":   0.13,
    "__label__soc_greeting":  0.10,
    "__label__soc_gratitude": 0.08,
    "__label__nav_command":   0.13,
    "__label__inf_question":  0.12,
    "__label__fdb_neg":       0.05,
    "__label__fdb_pos":       0.05,
    "__label__oos":           0.05,
}

# ── Invariant guards (explicit raises — safe under python -O, unlike assert) ──
if abs(sum(LABEL_WEIGHTS.values()) - 1.0) >= 1e-9:
    raise ValueError(
        f"LABEL_WEIGHTS must sum to 1.0, got {sum(LABEL_WEIGHTS.values()):.6f}"
    )

_missing_templates = set(LABEL_WEIGHTS) - set(TEMPLATES)
_missing_weights   = set(TEMPLATES)     - set(LABEL_WEIGHTS)
if _missing_templates:
    raise ValueError(f"No templates for labels: {_missing_templates}")
if _missing_weights:
    raise ValueError(f"No weight assigned for labels: {_missing_weights}")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PHONE NUMBER GENERATOR  (dynamic — prevents fixed-pattern overfitting)
# ══════════════════════════════════════════════════════════════════════════════

def random_phone(rng: random.Random) -> str:
    """
    Generate a plausible Ugandan mobile number.
    Prefixes: 070x (Airtel), 075x (Airtel), 077x (MTN), 078x (MTN), 076x (MTN).
    """
    prefix = rng.choice(["070", "075", "077", "078", "076"])
    digits = "".join(str(rng.randint(0, 9)) for _ in range(7))
    return prefix + digits


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SLOT FILLER
# ══════════════════════════════════════════════════════════════════════════════

def fill_slots(template: str, rng: random.Random) -> str:
    """
    Resolve all {slot} references in `template`.

    Slot catalogue:
      {amt_w}      — Luganda word form of a sampled amount  ("emitwalo kkumi")
      {amt_w_num}  — Digit string paired with {amt_w}       ("10000")
      {amt_n}      — Independently sampled numeric amount   ("7500")
      {person}     — A kinship / social role term
      {phone}      — Dynamically generated Ugandan phone number
      {provider}   — Mobile money / bank name
      {bill}       — Bill or fee type
      {account}    — Account descriptor (may contain sub-slot {provider})
      {ui}         — UI element name
      {item}       — General information noun
      {q}          — Question word
      {greet_g}    — General greeting phrase
      {greet_m}    — Morning greeting phrase
      {polite}     — Politeness particle
      {problem}    — Problem description phrase
      {pos_adj}    — Positive adjective phrase
      {neg_adj}    — Negative adjective phrase

    Raises ValueError on unknown slot names (catches template typos at
    generation time rather than silently producing malformed lines).
    """
    amt_pair = rng.choice(AMOUNTS)

    # Resolve {account} — one entry itself contains a {provider} sub-slot
    raw_account = rng.choice(ACCOUNTS)
    account_str = (
        raw_account.format(provider=rng.choice(PROVIDERS))
        if "{provider}" in raw_account
        else raw_account
    )

    slots: dict[str, str] = {
        "amt_w":     amt_pair.word,
        "amt_w_num": amt_pair.numeric,   # FIX: was dead data (amt_pair[1]); now exposed
        "amt_n":     rng.choice(NUMERIC_AMOUNTS),
        "person":    rng.choice(PEOPLE),
        "phone":     random_phone(rng),
        "provider":  rng.choice(PROVIDERS),
        "bill":      rng.choice(BILL_TYPES),
        "account":   account_str,
        "ui":        rng.choice(UI_ELEMENTS),
        "item":      rng.choice(ITEMS_INFO),
        "q":         rng.choice(Q_WORDS),
        "greet_g":   rng.choice(GREETINGS_GENERAL),
        "greet_m":   rng.choice(GREETINGS_MORNING),
        # greet_r removed — was defined but never referenced in any template
        "polite":    rng.choice(POLITENESS),
        "problem":   rng.choice(PROBLEMS),
        "pos_adj":   rng.choice(POSITIVE_ADJ),
        "neg_adj":   rng.choice(NEGATIVE_ADJ),
    }
    try:
        return template.format(**slots)
    except KeyError as exc:
        raise ValueError(
            f"Template references unknown slot {exc}: '{template}'"
        ) from exc


# ── Template pre-validation (runs once at import; surfaces typos immediately) ─
def _validate_templates() -> None:
    dummy_rng = random.Random(0)
    for label, tmpl_list in TEMPLATES.items():
        if label == "__label__oos":
            continue
        for tmpl in tmpl_list:
            try:
                fill_slots(tmpl, dummy_rng)
            except (ValueError, KeyError) as exc:
                raise AssertionError(
                    f"Bad template in {label}: {exc}\n  → '{tmpl}'"
                ) from exc

_validate_templates()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ELISION LAYER  (bidirectional via LugandaElisionHandler)
# ══════════════════════════════════════════════════════════════════════════════

def apply_elision(text: str, rng: random.Random,
                  probability: float = Config.ELISION_PROB) -> str:
    """
    Randomly applies one of three elision states:

      contracted (reconstruct_elisions) — formal texting style, e.g. "y'emmere"
      expanded   (expand_elisions)      — spoken/canonical style, e.g. "ya emmere"
      unchanged                         — plain/neutral form (~65 % of calls)

    LugandaElisionHandler is stateless and thread-safe after __init__.
    The probability budget is split 50/50 between the two active transforms,
    so each variant appears at rate (probability × 0.5) in training data.
    """
    if rng.random() > probability:
        return text

    if rng.random() < Config.ELISION_SPLIT:
        return _ELISION.reconstruct_elisions(text)
    else:
        return _ELISION.expand_elisions(text)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  NOISE INJECTION LAYER
# ══════════════════════════════════════════════════════════════════════════════

# Single combined alternation regex — O(n) replacement instead of O(15·n).
# Keys sorted longest-first so "50000" is never shadowed by a shorter prefix.
_SLANG_RE: re.Pattern = re.compile(
    r'\b(' + '|'.join(
        re.escape(k) for k in sorted(NUMERIC_SLANG, key=len, reverse=True)
    ) + r')\b'
)

# Pre-compiled per-word code-switch patterns (still used one-at-a-time for
# first-match semantics, but compiled at module load not per call).
_CODE_SWITCH_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(lug)}\b", re.IGNORECASE), eng)
    for lug, eng in CODE_SWITCH_MAP.items()
]


def inject_noise(text: str, rng: random.Random,
                 probability: float = Config.NOISE_PROB) -> str:
    """
    Simulate real-world dirty input with `probability` (default 20 %).

    Step order (important — code-switch runs BEFORE lowercase so that English
    replacements are not left title-cased inside an otherwise all-lowercase line):

      1. Numeric slang   — "5000" → "5k"          (always, when numbers present)
      2. Code-switch     — Luganda verb → English   (20 % sub-chance, one word)
      3. Lowercase       — "Sindiika" → "sindiika"  (30 % sub-chance)
      4. Punct drop      — strip trailing ".?!"      (30 % sub-chance)
    """
    if rng.random() > probability:
        return text

    # 1. Numeric slang (single-pass combined regex)
    text = _SLANG_RE.sub(lambda m: NUMERIC_SLANG[m.group(0)], text)

    # 2. Code-switch one verb BEFORE lowercasing (avoids mixed-case artefact)
    if rng.random() < Config.CODE_SWITCH_PROB:
        for pattern, english_word in _CODE_SWITCH_PATTERNS:
            replaced = pattern.sub(english_word, text, count=1)
            if replaced != text:
                text = replaced
                break

    # 3. Lowercase (applied after code-switch so casing is consistent)
    if rng.random() < Config.LOWERCASE_PROB:
        text = text.lower()

    # 4. Drop trailing punctuation
    if rng.random() < Config.PUNCT_DROP_PROB:
        text = text.rstrip(".?!")

    return text


# ══════════════════════════════════════════════════════════════════════════════
# 7.  GENERATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_for_label(label: str, quota: int, rng: random.Random) -> list[str]:
    """
    Generate exactly `quota` unique lines for one label.

    OOS lines are verbatim strings — elision and noise are skipped to avoid
    corrupting English/nonsense text with Luganda morphological transforms.
    OOS is handled via rng.sample() which is O(quota) and guarantees uniqueness
    without any retry loop, while also enforcing quota ≤ pool size at runtime.
    """
    templates = TEMPLATES[label]

    # ── Fast path: OOS verbatim strings ───────────────────────────────────────
    if label == "__label__oos":
        pool = [f"{label} {t}" for t in templates]
        if quota > len(pool):
            raise RuntimeError(
                f"[{label}] OOS quota ({quota}) exceeds the number of unique "
                f"verbatim templates ({len(pool)}).\n"
                f"  → Add at least {quota - len(pool)} more OOS template(s)."
            )
        return rng.sample(pool, quota)

    # ── Normal path: slot-filled + elision + noise ────────────────────────────
    bucket:      set[str] = set()
    max_attempts = quota * Config.MAX_ATTEMPT_MULT
    stall_limit  = min(Config.STALL_LIMIT_CAP, quota * Config.STALL_LIMIT_MULT)
    attempts     = 0
    stall        = 0

    while len(bucket) < quota and attempts < max_attempts:
        attempts += 1
        prev_size = len(bucket)

        template = rng.choice(templates)
        text     = fill_slots(template, rng)
        text     = apply_elision(text, rng)
        text     = inject_noise(text, rng)
        bucket.add(f"{label} {text}")

        if len(bucket) == prev_size:
            stall += 1
            if stall >= stall_limit:
                break        # collision ceiling reached — fail fast
        else:
            stall = 0

    if len(bucket) < quota:
        raise RuntimeError(
            f"[{label}] Generated only {len(bucket)}/{quota} unique lines "
            f"after {attempts} attempts (stall_limit={stall_limit}).\n"
            f"  → Add more templates or expand lexical variety for this label."
        )
    return list(bucket)


def _compute_quotas(target_count: int) -> dict[str, int]:
    """
    Compute per-label line counts that sum exactly to `target_count` using the
    largest-remainder (Hamilton) method.  Returns a dict[label → count] to make
    the label↔quota mapping explicit and immune to iteration-order assumptions.
    """
    labels  = list(LABEL_WEIGHTS.keys())
    weights = list(LABEL_WEIGHTS.values())
    raw     = [w * target_count for w in weights]
    quotas  = [int(r) for r in raw]
    by_frac = sorted(range(len(labels)), key=lambda i: -(raw[i] - quotas[i]))
    for i in by_frac[: target_count - sum(quotas)]:
        quotas[i] += 1
    return dict(zip(labels, quotas))


def generate_dataset(target_count: int = 1200, seed: int = 42) -> list[str]:
    rng    = random.Random(seed)
    quotas = _compute_quotas(target_count)   # dict — no positional coupling

    all_lines: list[str] = []
    for label, quota in quotas.items():
        print(f"  generating {quota:>4} lines  {label} …", flush=True)
        all_lines.extend(generate_for_label(label, quota, rng))

    return all_lines


# ══════════════════════════════════════════════════════════════════════════════
# 8.  QUALITY METRICS  (single pass — avoids iterating data 5+ times)
# ══════════════════════════════════════════════════════════════════════════════

def print_quality_report(data: list[str]) -> None:
    n               = len(data)
    vocab:  set[str] = set()
    total_tokens     = 0
    label_counts:   Counter = Counter()
    elided           = 0
    slangy           = 0
    switched         = 0
    slang_values     = set(NUMERIC_SLANG.values())
    switch_values    = set(CODE_SWITCH_MAP.values())

    for line in data:
        label, _, text = line.partition(" ")
        label_counts[label] += 1
        tokens = text.lower().split()
        vocab.update(tokens)
        total_tokens += len(tokens)
        if "'" in text:
            elided += 1
        if any(s in text for s in slang_values):
            slangy += 1
        if any(e in text for e in switch_values):
            switched += 1

    avg_len     = total_tokens / n
    entropy     = -sum((c / n) * math.log2(c / n) for c in label_counts.values())
    max_entropy = math.log2(len(label_counts))

    print(f"\n  Vocabulary size       : {len(vocab):,} unique tokens")
    print(f"  Avg sentence length   : {avg_len:.1f} tokens")
    print(
        f"  Label entropy         : {entropy:.3f} bits  "
        f"(max = {max_entropy:.3f} for {len(label_counts)} classes)"
    )
    print("\n  Label distribution:")
    for lbl, count in sorted(label_counts.items()):
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        print(f"    {lbl:<32}  {count:>4}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Elision coverage      : {elided:>4} lines  ({elided/n*100:.1f}%)")
    print(f"  Numeric slang         : {slangy:>4} lines  ({slangy/n*100:.1f}%)")
    print(f"  Code-switched         : {switched:>4} lines  ({switched/n*100:.1f}%)")

    dupes  = n - len(set(data))
    status = "✅" if dupes == 0 else "⚠️ "
    print(f"\n  {status} Duplicates          : {dupes}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Luganda FastText training data generator v2.1"
    )
    parser.add_argument(
        "--target", type=int, default=1200,
        help="Total number of training lines to generate (default: 1200)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for full reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output", default="luganda_train.txt",
        help="Output file path (default: luganda_train.txt)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    TARGET = args.target
    SEED   = args.seed
    OUTPUT = args.output

    print(f"Luganda Dataset Factory v2.1  —  target={TARGET}, seed={SEED}\n")
    data = generate_dataset(TARGET, SEED)

    # Final shuffle with a different seed to decorrelate from per-label order
    rng = random.Random(SEED + 1)
    rng.shuffle(data)

    try:
        with open(OUTPUT, "w", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")
    except OSError as exc:
        raise RuntimeError(
            f"Failed to write output to '{OUTPUT}': {exc}"
        ) from exc

    print(f"\n✅ Written {len(data)} lines → {OUTPUT}")
    print_quality_report(data)
    print(
        "\nNext step:\n"
        f"  fasttext supervised -input {OUTPUT} -output model_luganda \\\n"
        "    -wordNgrams 2 -minCount 1 -epoch 25 -lr 0.5\n"
    )
