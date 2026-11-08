"""Which language Jarvis speaks, and which language reminders use.

Settings:
  reply_language     "auto" (match the user) or a code from LANGUAGES
  reminder_language  "same" (as Jarvis) or a code from LANGUAGES ("auto" works too)

In "auto", the language of the user's recent messages is noticed (state "user_lang") so pop-ups can use it.
"""

from __future__ import annotations

import re

# code -> (menu label, how to describe it to the model)
LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("English", "English"),
    "hinglish": ("Hinglish", "Hinglish - Hindi written and spoken in everyday Roman letters, mixed naturally with "
                             "English words (e.g. 'Haan, main abhi kar deta hoon')"),
    "hi": ("Hindi · हिन्दी", "Hindi (simple everyday Hindi; write it in Devanagari)"),
    "ur": ("Urdu · اردو", "Urdu (write it in Urdu script)"),
    "mr": ("Marathi · मराठी", "Marathi"),
    "bn": ("Bengali · বাংলা", "Bengali"),
    "gu": ("Gujarati · ગુજરાતી", "Gujarati"),
    "pa": ("Punjabi · ਪੰਜਾਬੀ", "Punjabi (Gurmukhi script)"),
    "ta": ("Tamil · தமிழ்", "Tamil"),
    "te": ("Telugu · తెలుగు", "Telugu"),
    "es": ("Spanish · Español", "Spanish"),
    "fr": ("French · Français", "French"),
    "de": ("German · Deutsch", "German"),
    "ar": ("Arabic · العربية", "Arabic"),
}
AUTO_LABEL = "Auto - match how I talk"
SAME_LABEL = "Same as Jarvis"
ALIASES = {
    "english": "en", "angrezi": "en", "inglish": "en", "hinglish": "hinglish", "hindi": "hi", "हिंदी": "hi",
    "हिन्दी": "hi", "urdu": "ur", "اردو": "ur", "marathi": "mr", "मराठी": "mr", "bengali": "bn", "bangla": "bn",
    "বাংলা": "bn", "gujarati": "gu", "ગુજરાતી": "gu", "punjabi": "pa", "ਪੰਜਾਬੀ": "pa", "tamil": "ta",
    "தமிழ்": "ta", "telugu": "te", "తెలుగు": "te", "spanish": "es", "español": "es", "espanol": "es",
    "french": "fr", "français": "fr", "francais": "fr", "german": "de", "deutsch": "de", "arabic": "ar",
    "العربية": "ar", "auto": "auto", "automatic": "auto", "same": "same",
}

# Words on the reminder pop-ups.
_KEYS = ("reminder", "reminder_set", "missed", "daily", "weekdays", "due_now", "due_while_off", "done", "snooze",
         "snooze_for", "minutes", "hour", "cancel", "under_min", "in_min", "in_h", "in_hm", "in_days", "hide")
_WORDS = {
    "en": ("REMINDER", "REMINDER SET", "MISSED", "DAILY", "WEEKDAYS", "Your task is due now",
           "Was due while Jarvis was off", "Done", "Snooze 10 min", "Snooze for…", "{n} minutes", "1 hour",
           "Cancel reminder", "in under a minute", "in {n} min", "in {h} h", "in {h} h {m} min", "in {d} days", "Hide"),
    "hinglish": ("REMINDER", "REMINDER SET", "MISSED", "ROZ", "MON–FRI", "Aapka task ab karna hai",
                 "Jarvis band tha tab ka reminder", "Ho gaya", "10 min baad", "Kitni der baad…", "{n} minute",
                 "1 ghanta", "Reminder hatao", "1 minute se kam me", "{n} min me", "{h} ghante me",
                 "{h} ghante {m} min me", "{d} din me", "Chhupao"),
    "hi": ("रिमाइंडर", "रिमाइंडर सेट", "छूटा हुआ", "रोज़", "सोम–शुक्र", "आपका काम अब करना है",
           "Jarvis बंद था तब का रिमाइंडर", "हो गया", "10 मिनट बाद", "कितनी देर बाद…", "{n} मिनट", "1 घंटा",
           "रिमाइंडर हटाएँ", "1 मिनट से कम में", "{n} मिनट में", "{h} घंटे में", "{h} घंटे {m} मिनट में",
           "{d} दिन में", "छुपाएँ"),
    "ur": ("یاد دہانی", "یاد دہانی سیٹ", "چھوٹ گئی", "روزانہ", "پیر تا جمعہ", "آپ کا کام اب کرنا ہے",
           "Jarvis بند تھا تب کی یاد دہانی", "ہو گیا", "10 منٹ بعد", "کتنی دیر بعد…", "{n} منٹ", "1 گھنٹہ",
           "یاد دہانی ہٹائیں", "1 منٹ سے کم میں", "{n} منٹ میں", "{h} گھنٹے میں", "{h} گھنٹے {m} منٹ میں",
           "{d} دن میں", "چھپائیں"),
    "mr": ("रिमाइंडर", "रिमाइंडर सेट", "चुकलेले", "रोज", "सोम–शुक्र", "तुमचे काम आता करायचे आहे",
           "Jarvis बंद असताना आलेले", "झाले", "10 मिनिटांनी", "किती वेळाने…", "{n} मिनिटे", "1 तास",
           "रिमाइंडर रद्द करा", "1 मिनिटापेक्षा कमी वेळात", "{n} मिनिटांत", "{h} तासांत", "{h} तास {m} मिनिटांत",
           "{d} दिवसांत", "लपवा"),
    "bn": ("রিমাইন্ডার", "রিমাইন্ডার সেট", "মিস হয়েছে", "রোজ", "সোম–শুক্র", "আপনার কাজের সময় হয়েছে",
           "Jarvis বন্ধ থাকাকালীন", "হয়ে গেছে", "10 মিনিট পরে", "কতক্ষণ পরে…", "{n} মিনিট", "1 ঘণ্টা",
           "রিমাইন্ডার বাতিল", "1 মিনিটের কম সময়ে", "{n} মিনিটে", "{h} ঘণ্টায়", "{h} ঘণ্টা {m} মিনিটে",
           "{d} দিনে", "লুকান"),
    "gu": ("રિમાઇન્ડર", "રિમાઇન્ડર સેટ", "ચૂકી ગયા", "રોજ", "સોમ–શુક્ર", "તમારું કામ હવે કરવાનું છે",
           "Jarvis બંધ હતું ત્યારે", "થઈ ગયું", "10 મિનિટ પછી", "કેટલી વાર પછી…", "{n} મિનિટ", "1 કલાક",
           "રિમાઇન્ડર રદ કરો", "1 મિનિટથી ઓછામાં", "{n} મિનિટમાં", "{h} કલાકમાં", "{h} કલાક {m} મિનિટમાં",
           "{d} દિવસમાં", "છુપાવો"),
    "pa": ("ਰਿਮਾਈਂਡਰ", "ਰਿਮਾਈਂਡਰ ਸੈੱਟ", "ਖੁੰਝ ਗਿਆ", "ਰੋਜ਼", "ਸੋਮ–ਸ਼ੁੱਕਰ", "ਤੁਹਾਡਾ ਕੰਮ ਹੁਣ ਕਰਨਾ ਹੈ",
           "Jarvis ਬੰਦ ਸੀ ਉਦੋਂ ਦਾ", "ਹੋ ਗਿਆ", "10 ਮਿੰਟ ਬਾਅਦ", "ਕਿੰਨੀ ਦੇਰ ਬਾਅਦ…", "{n} ਮਿੰਟ", "1 ਘੰਟਾ",
           "ਰਿਮਾਈਂਡਰ ਹਟਾਓ", "1 ਮਿੰਟ ਤੋਂ ਘੱਟ ਵਿੱਚ", "{n} ਮਿੰਟ ਵਿੱਚ", "{h} ਘੰਟੇ ਵਿੱਚ", "{h} ਘੰਟੇ {m} ਮਿੰਟ ਵਿੱਚ",
           "{d} ਦਿਨਾਂ ਵਿੱਚ", "ਲੁਕਾਓ"),
    "ta": ("நினைவூட்டல்", "நினைவூட்டல் அமைந்தது", "தவறியது", "தினமும்", "திங்கள்–வெள்ளி",
           "உங்கள் வேலைக்கு நேரம் ஆனது", "Jarvis மூடியிருந்தபோது", "முடிந்தது", "10 நிமிடம் கழித்து",
           "எவ்வளவு நேரம் கழித்து…", "{n} நிமிடங்கள்", "1 மணி நேரம்", "நினைவூட்டலை ரத்து செய்",
           "1 நிமிடத்திற்குள்", "{n} நிமிடத்தில்", "{h} மணி நேரத்தில்", "{h} மணி {m} நிமிடத்தில்",
           "{d} நாட்களில்", "மறை"),
    "te": ("రిమైండర్", "రిమైండర్ సెట్ అయింది", "మిస్ అయింది", "రోజూ", "సోమ–శుక్ర", "మీ పని చేసే సమయం వచ్చింది",
           "Jarvis ఆఫ్‌లో ఉన్నప్పుడు", "అయిపోయింది", "10 నిమిషాల తర్వాత", "ఎంతసేపు తర్వాత…", "{n} నిమిషాలు",
           "1 గంట", "రిమైండర్ రద్దు చేయి", "1 నిమిషంలోపు", "{n} నిమిషాల్లో", "{h} గంటల్లో",
           "{h} గం {m} ని.లో", "{d} రోజుల్లో", "దాచు"),
    "es": ("RECORDATORIO", "RECORDATORIO LISTO", "PERDIDO", "DIARIO", "LUN–VIE", "Es hora de tu tarea",
           "Venció mientras Jarvis estaba apagado", "Hecho", "Posponer 10 min", "Posponer…", "{n} minutos",
           "1 hora", "Cancelar recordatorio", "en menos de un minuto", "en {n} min", "en {h} h", "en {h} h {m} min",
           "en {d} días", "Ocultar"),
    "fr": ("RAPPEL", "RAPPEL CRÉÉ", "MANQUÉ", "CHAQUE JOUR", "LUN–VEN", "C'est l'heure de ta tâche",
           "Prévu pendant que Jarvis était fermé", "Fait", "Reporter 10 min", "Reporter de…", "{n} minutes",
           "1 heure", "Annuler le rappel", "dans moins d'une minute", "dans {n} min", "dans {h} h",
           "dans {h} h {m} min", "dans {d} jours", "Masquer"),
    "de": ("ERINNERUNG", "ERINNERUNG GESETZT", "VERPASST", "TÄGLICH", "MO–FR", "Zeit für deine Aufgabe",
           "Fällig, während Jarvis aus war", "Erledigt", "10 Min später", "Später um…", "{n} Minuten", "1 Stunde",
           "Erinnerung löschen", "in unter einer Minute", "in {n} Min", "in {h} Std", "in {h} Std {m} Min",
           "in {d} Tagen", "Ausblenden"),
    "ar": ("تذكير", "تم ضبط التذكير", "فائت", "يومي", "أيام العمل", "حان وقت مهمتك", "كان مستحقًا أثناء إغلاق Jarvis",
           "تم", "غفوة 10 دقائق", "غفوة لمدة…", "{n} دقيقة", "ساعة واحدة", "إلغاء التذكير", "خلال أقل من دقيقة",
           "خلال {n} دقيقة", "خلال {h} ساعة", "خلال {h} ساعة و{m} دقيقة", "خلال {d} أيام", "إخفاء"),
}
STRINGS = {code: dict(zip(_KEYS, words, strict=True)) for code, words in _WORDS.items()}
RTL = {"ur", "ar"}


def words(lang: str) -> dict[str, str]:
    return STRINGS.get(lang, STRINGS["en"])


def label(code: str) -> str:
    if code == "auto":
        return AUTO_LABEL
    if code == "same":
        return SAME_LABEL
    return LANGUAGES.get(code, LANGUAGES["en"])[0]


def find(name: str) -> str | None:
    """'Hindi', 'हिंदी', 'hi', 'hinglish' -> a code; None when unknown."""
    n = (name or "").strip().lower()
    if n in LANGUAGES or n in ("auto", "same"):
        return n
    if n in ALIASES:
        return ALIASES[n]
    for word in re.findall(r"[^\W\d_]+", n):  # "hindi me", "in hindi please"
        if word in ALIASES:
            return ALIASES[word]
    return None


# -- noticing the user's language ---------------------------------------------------------------
_SCRIPTS = (("hi", "ऀ", "ॿ"), ("bn", "ঀ", "৿"), ("pa", "਀", "੿"),
            ("gu", "઀", "૿"), ("ta", "஀", "௿"), ("te", "ఀ", "౿"),
            ("ur", "؀", "ۿ"))
_HINGLISH = {"hai", "hain", "kar", "karo", "kardo", "krdo", "karna", "karni", "mujhe", "mera", "meri", "mere",
             "kya", "kyu", "kyun", "nahi", "nhi", "ka", "ke", "ki", "ko", "aur", "bhi", "aaj", "yaar", "bhai",
             "haan", "theek", "thik", "acha", "achha", "wala", "wali", "baje", "dena", "dilana", "tum", "aap",
             "apna", "abhi", "kaise", "kaun", "kab", "hoga", "tha", "thi", "se", "mein", "liye", "batao",
             "chalo", "kuch", "dekho", "jao", "hoon", "hu", "raha", "rahi", "gaya", "gayi", "wahi", "yeh", "ye",
             "woh", "wo", "jab", "tab", "pehle", "baad", "bahut", "bohot", "zara", "lekin"}  # not "me"/"do"/"so"
_URDU_ONLY = set("ےٹڈڑںھ")


def detect(text: str) -> str | None:
    """A best guess at the language of one message; None when it's too short to tell."""
    t = (text or "").strip()
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 4:
        return None
    for code, lo, hi in _SCRIPTS:
        n = sum(1 for c in letters if lo <= c <= hi)
        if n >= max(3, len(letters) * 0.3):
            if code == "ur" and not (set(t) & _URDU_ONLY):
                return "ar"
            return code
    tokens = re.findall(r"[a-z]+", t.lower())
    if len(tokens) < 2:
        return None
    hits = sum(1 for w in tokens if w in _HINGLISH)
    if hits >= 2 or hits / len(tokens) >= 0.34:
        return "hinglish"
    return "en"


def remember(state, text: str) -> None:
    """Called for each user message: keep the latest clear guess (only written when it changes)."""
    if state is None:
        return
    guess = detect(text)
    if guess and state.get("user_lang") != guess:
        state.set("user_lang", guess)


# -- resolving the settings ---------------------------------------------------------------------
def reply_language(settings, state=None) -> str:
    """The language Jarvis speaks now (auto -> the user's language, English when unknown)."""
    code = getattr(settings, "reply_language", "auto") or "auto"
    if code == "auto":
        return (state.get("user_lang") if state is not None else None) or "en"
    return code if code in LANGUAGES else "en"


def reminder_language(settings, state=None) -> str:
    code = getattr(settings, "reminder_language", "same") or "same"
    if code == "same":
        return reply_language(settings, state)
    if code == "auto":
        return (state.get("user_lang") if state is not None else None) or "en"
    return code if code in LANGUAGES else "en"


def prompt_rules(settings) -> str:
    """Lines for the system prompt."""
    reply = getattr(settings, "reply_language", "auto") or "auto"
    rem = getattr(settings, "reminder_language", "same") or "same"
    if reply in LANGUAGES:
        lines = [f"- Always reply in {LANGUAGES[reply][1]} - the user chose this in Settings - even when they write "
                 "in another language. Names, apps and technical words may stay in English. If they ask to talk in "
                 "another language, call set_language."]
    else:
        lines = ["- Reply in the language the user speaks (English, Hindi, Urdu, Hinglish ...). Match their style "
                 "and slang. If they ask you to always use one language, call set_language."]
    if rem in LANGUAGES:
        lines.append(f"- Reminders are in {LANGUAGES[rem][0].split(' · ')[0]} (the user's choice): write the "
                     "set_reminder message in it, and say ringing reminders in it.")
    elif rem == "auto" and reply in LANGUAGES:
        lines.append("- Reminders follow the language the user speaks: write the set_reminder message and say "
                     "ringing reminders in the language they used.")
    return "\n".join(lines)
