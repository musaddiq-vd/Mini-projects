"""Built-in English practice content (works offline). Each MCQ: (question, options, answer index, why).

Levels: 1 = beginner, 2 = intermediate, 3 = advanced.
"""

from __future__ import annotations

# -- grammar & sentence structure -------------------------------------------------------
GRAMMAR = [
    (1, "She ___ to school every day.", ["go", "goes", "going", "gone"], 1,
     "With he/she/it in the present simple, add -s: she goes."),
    (1, "I ___ a student.", ["is", "are", "am", "be"], 2, "'I' always takes 'am'."),
    (1, "They ___ playing cricket now.", ["is", "are", "am", "was"], 1, "'They' takes 'are'; 'now' needs the present continuous."),
    (1, "Choose the correct sentence.", ["He don't like tea.", "He doesn't likes tea.", "He doesn't like tea.",
                                         "He not like tea."], 2, "Negative present simple: doesn't + base verb (like)."),
    (1, "There ___ two books on the table.", ["is", "are", "be", "has"], 1, "Plural noun (two books) -> 'there are'."),
    (1, "___ you like coffee?", ["Does", "Do", "Are", "Is"], 1, "Questions with 'you' in present simple use 'Do'."),
    (1, "I have ___ apple.", ["a", "an", "the", "-"], 1, "'an' before a vowel sound: an apple."),
    (1, "Yesterday I ___ to the market.", ["go", "goes", "went", "gone"], 2, "'Yesterday' -> past simple: went."),
    (1, "Choose the correct word order.", ["Always I drink tea.", "I drink always tea.", "I always drink tea.",
                                           "I drink tea always."], 2, "Frequency adverbs go before the main verb."),
    (1, "My brother is ___ than me.", ["tall", "taller", "tallest", "more tall"], 1, "Comparing two -> comparative: taller."),
    (2, "I ___ here since 2020.", ["live", "am living", "have lived", "lived"], 2,
     "'since' + a starting point -> present perfect: have lived."),
    (2, "If it rains, we ___ at home.", ["stay", "will stay", "would stay", "stayed"], 1,
     "First conditional: if + present, will + verb."),
    (2, "She asked me where ___.", ["do I live", "I live", "I lived", "did I live"], 2,
     "Reported questions use statement order and shift tense: where I lived."),
    (2, "The report ___ by the manager yesterday.", ["wrote", "was written", "is written", "has written"], 1,
     "Passive past: was/were + past participle."),
    (2, "I'm looking forward ___ you.", ["to meet", "meeting", "to meeting", "meet"], 2,
     "'look forward to' is followed by -ing: to meeting."),
    (2, "He's good ___ solving problems.", ["in", "at", "on", "for"], 1, "We are good AT something."),
    (2, "Choose the correct sentence.", ["I am agree with you.", "I agree with you.", "I am agreeing you.",
                                         "I agreed with you now."], 1, "'agree' is a verb; no 'am' needed."),
    (2, "By the time we arrived, the movie ___.", ["started", "has started", "had started", "starts"], 2,
     "An earlier past action -> past perfect: had started."),
    (2, "She has worked here ___ five years.", ["since", "for", "from", "during"], 1, "'for' + a length of time."),
    (2, "Neither my friends nor my brother ___ coming.", ["are", "is", "were", "be"], 1,
     "With neither...nor, the verb agrees with the nearer subject (brother -> is)."),
    (2, "Which is the most natural sentence?", ["Please do the needful.", "Kindly revert back soon.",
                                                "Could you reply when you get a chance?", "Do one thing, call me."], 2,
     "Polite, natural English: 'Could you reply when you get a chance?'"),
    (3, "Hardly ___ the room when the phone rang.", ["I had entered", "had I entered", "I entered", "did I entered"], 1,
     "Negative adverbs at the start invert subject and auxiliary: Hardly had I..."),
    (3, "If I ___ you, I would take the job.", ["am", "was", "were", "be"], 2, "Second conditional: 'If I were you'."),
    (3, "She would have passed if she ___ harder.", ["studied", "had studied", "would study", "has studied"], 1,
     "Third conditional: if + past perfect, would have + participle."),
    (3, "It's high time we ___.", ["leave", "left", "are leaving", "will leave"], 1, "'It's high time' + past simple."),
    (3, "The candidate, ___ experience impressed us, was hired.", ["who", "whom", "whose", "which"], 2,
     "Possession -> whose."),
    (3, "Not only ___ late, but he also forgot the files.", ["he was", "was he", "he is", "is he"], 1,
     "'Not only' at the start -> inversion: was he."),
    (3, "I'd rather you ___ smoke here.", ["don't", "didn't", "won't", "not"], 1, "'I'd rather you' + past simple."),
    (3, "Choose the most concise sentence.", ["In my personal opinion, I think it's good.", "I think it's good.",
                                              "Personally, in my opinion, it's good.", "My opinion I think is good."], 1,
     "Avoid repeating 'personal', 'opinion', 'I think'."),
    (3, "The data ___ that sales increased.", ["suggest", "suggests", "both are acceptable", "suggesting"], 2,
     "'Data' can be plural (formal) or singular (common usage)."),
]

# -- vocabulary ------------------------------------------------------------------------
VOCAB = [
    (1, "What is the opposite of 'cheap'?", ["expensive", "poor", "small", "easy"], 0, "cheap <-> expensive."),
    (1, "Which word means 'very tired'?", ["sleepy", "exhausted", "bored", "lazy"], 1, "exhausted = extremely tired."),
    (1, "Choose the synonym of 'begin'.", ["end", "start", "stop", "finish"], 1, "begin = start."),
    (1, "A place where you borrow books is a ___.", ["bookstore", "library", "classroom", "office"], 1,
     "You borrow from a library; you buy at a bookstore."),
    (1, "'Delicious' describes ___.", ["music", "food", "weather", "a car"], 1, "delicious = tastes very good."),
    (1, "Which is a polite way to ask?", ["Give me water.", "Water!", "Could I have some water, please?",
                                          "I want water now."], 2, "Could I have..., please? is polite."),
    (2, "'Punctual' means ___.", ["always on time", "very careful", "hard-working", "friendly"], 0,
     "punctual = arrives on time."),
    (2, "Choose the word for 'to make something better'.", ["improve", "remove", "approve", "prove"], 0,
     "improve = make better."),
    (2, "I need to ___ an appointment with the doctor.", ["do", "make", "take", "give"], 1, "Collocation: make an appointment."),
    (2, "'Reliable' person is someone you can ___.", ["ignore", "trust", "avoid", "forget"], 1, "reliable = trustworthy."),
    (2, "Which phrasal verb means 'to postpone'?", ["put off", "put on", "put up", "put in"], 0, "put off = delay."),
    (2, "Choose the best word: The meeting was very ___; we decided a lot.", ["productive", "product", "producer",
                                                                              "productively"], 0,
     "An adjective is needed: productive."),
    (3, "'Meticulous' means ___.", ["careless", "very careful and precise", "quick", "talkative"], 1,
     "meticulous = paying great attention to detail."),
    (3, "A 'pragmatic' approach is ___.", ["practical", "emotional", "theoretical", "risky"], 0,
     "pragmatic = practical, realistic."),
    (3, "Which word fits: She gave a ___ answer, avoiding a clear yes or no.", ["candid", "evasive", "blunt",
                                                                                 "concise"], 1,
     "evasive = avoiding a direct answer."),
    (3, "'To leverage' something means to ___.", ["ignore it", "use it to your advantage", "lift it", "sell it"], 1,
     "leverage = use something to get the most benefit."),
    (3, "Choose the idiom meaning 'very easy'.", ["a piece of cake", "break the ice", "hit the sack",
                                                   "under the weather"], 0, "a piece of cake = very easy."),
    (3, "'Under the weather' means ___.", ["outside", "a little ill", "very happy", "in the rain"], 1,
     "under the weather = feeling slightly sick."),
]

# -- reading: (level, title, passage, [mcq...]) -----------------------------------------------
READING = [
    (1, "Riya's Morning", "Riya wakes up at six o'clock every day. She drinks a glass of warm water and goes for a "
     "short walk in the park. After her walk, she takes a shower and eats breakfast with her family. Her favourite "
     "breakfast is poha with tea. At eight thirty, she takes the bus to her office.",
     [("What time does Riya wake up?", ["5:00", "6:00", "7:00", "8:30"], 1, "She wakes up at six o'clock."),
      ("What does she do after her walk?", ["Eats lunch", "Takes a shower", "Goes to sleep", "Reads"], 1,
       "After the walk she takes a shower."),
      ("How does she go to the office?", ["By car", "On foot", "By bus", "By train"], 2, "She takes the bus.")]),
    (1, "A New Phone", "Aman saved money for six months to buy a new phone. Last Sunday he went to the shop with his "
     "sister. The phone he wanted was out of stock, so he chose a blue one with a bigger battery. He was very happy "
     "because the battery lasts two days.",
     [("How long did Aman save money?", ["Two months", "Six months", "One year", "Two days"], 1, "Six months."),
      ("Why didn't he buy the phone he wanted?", ["Too expensive", "Out of stock", "Wrong colour", "Shop closed"], 1,
       "It was out of stock."),
      ("Why was he happy?", ["The phone was free", "The battery lasts long", "His sister paid", "It was red"], 1,
       "The battery lasts two days.")]),
    (2, "Working From Home", "Many companies now let employees work from home a few days a week. Supporters say it "
     "saves travel time and helps people focus. Critics worry that teams lose the quick conversations that spark new "
     "ideas. A common solution is a hybrid plan: people meet in the office for team work and stay home for deep, "
     "focused tasks.",
     [("What is one benefit of working from home?", ["More meetings", "Saves travel time", "Bigger offices",
                                                     "Less focus"], 1, "It saves travel time."),
      ("What do critics worry about?", ["Higher salaries", "Losing quick conversations", "Too much travel",
                                         "Bad internet"], 1, "They worry teams lose quick conversations."),
      ("What is a 'hybrid plan'?", ["Only office work", "Only home work", "A mix of office and home",
                                    "Working at night"], 2, "Hybrid = a mix.")]),
    (2, "The Power of Sleep", "Scientists say adults need seven to nine hours of sleep. During sleep, the brain stores "
     "memories and the body repairs itself. People who sleep too little often find it hard to concentrate and may get "
     "ill more often. Experts suggest keeping the same bedtime and avoiding screens an hour before sleep.",
     [("How much sleep do adults need?", ["4-5 hours", "5-6 hours", "7-9 hours", "10-12 hours"], 2, "7 to 9 hours."),
      ("What does the brain do during sleep?", ["Stops working", "Stores memories", "Forgets everything",
                                                 "Grows bigger"], 1, "It stores memories."),
      ("What do experts suggest?", ["Use screens in bed", "Change bedtime daily", "Avoid screens before sleep",
                                    "Sleep less"], 2, "Avoid screens an hour before sleep.")]),
    (3, "Learning a Language", "Research suggests that adults can learn languages efficiently when practice is "
     "frequent, varied and meaningful. Short daily sessions tend to outperform occasional long ones, because spaced "
     "repetition strengthens memory. Equally important is 'comprehensible input': material that is slightly above the "
     "learner's level. Finally, speaking early, even with mistakes, builds confidence and fluency faster than waiting "
     "for perfection.",
     [("Which practice works better?", ["Occasional long sessions", "Short daily sessions", "No practice",
                                        "Only grammar books"], 1, "Short daily sessions."),
      ("What is 'comprehensible input'?", ["Very easy material", "Material slightly above your level",
                                           "Only native films", "Grammar rules"], 1,
       "Material slightly above the learner's level."),
      ("What does the writer say about mistakes?", ["Avoid speaking until perfect", "Speaking early helps",
                                                    "Mistakes stop progress", "Only write"], 1,
       "Speaking early, even with mistakes, builds fluency.")]),
    (3, "The Gig Economy", "The gig economy refers to short-term, flexible jobs, often found through apps. For workers "
     "it offers freedom to choose hours and projects, but income can be unpredictable and benefits such as health "
     "insurance are rare. Governments in several countries are debating new rules to protect gig workers without "
     "removing the flexibility that attracts them.",
     [("What is the gig economy?", ["Long-term office jobs", "Short-term flexible jobs", "Music concerts",
                                    "Government jobs"], 1, "Short-term, flexible jobs."),
      ("What is a disadvantage for gig workers?", ["Too much freedom", "Unpredictable income", "Too many benefits",
                                                   "Fixed hours"], 1, "Income can be unpredictable."),
      ("What are governments trying to do?", ["Ban apps", "Protect workers but keep flexibility",
                                              "Remove flexibility", "Raise taxes only"], 1,
       "Protect workers without removing flexibility.")]),
]

# -- listening: Jarvis reads the script aloud, then questions ---------------------------------
LISTENING = [
    (1, "At the Café", "Hi, can I get a cappuccino and a chocolate muffin, please? Oh, and could you make the coffee "
     "with less sugar? Thanks. How much is that? Two hundred and fifty rupees? Sure, I'll pay by card.",
     [("What drink did the person order?", ["Tea", "Cappuccino", "Juice", "Cold coffee"], 1, "A cappuccino."),
      ("What did they ask to change?", ["More milk", "Less sugar", "Bigger cup", "No ice"], 1, "Less sugar."),
      ("How did they pay?", ["Cash", "Card", "UPI", "They didn't pay"], 1, "By card.")]),
    (1, "A Phone Call", "Hello, this is Sameer. I'm calling about the flat for rent. Is it still available? Great. Can I "
     "come and see it tomorrow at five in the evening? Perfect, see you then.",
     [("Why is Sameer calling?", ["To buy a car", "About a flat for rent", "To book a table", "To complain"], 1,
       "About a flat for rent."),
      ("When will he visit?", ["Today at five", "Tomorrow at five", "Tomorrow morning", "Next week"], 1,
       "Tomorrow at five in the evening."),
      ("Is the flat available?", ["Yes", "No", "Not sure", "Only next month"], 0, "He says 'Great', so yes.")]),
    (2, "Team Update", "Good morning, everyone. Quick update: the client loved the new design, but they want the launch "
     "moved from the fifth to the twelfth. That gives us one more week for testing. Priya, please share the bug list "
     "by Thursday, and let's meet again on Friday at eleven.",
     [("What did the client think of the design?", ["Didn't like it", "Loved it", "Wanted it changed",
                                                     "Didn't see it"], 1, "The client loved it."),
      ("What is the new launch date?", ["The fifth", "The tenth", "The twelfth", "Friday"], 2, "The twelfth."),
      ("What must Priya do by Thursday?", ["Call the client", "Share the bug list", "Book a room", "Launch"], 1,
       "Share the bug list.")]),
    (2, "Weather Report", "Good evening. Tomorrow will start cloudy in the morning, with light rain around noon. The "
     "afternoon will clear up, and temperatures will reach twenty-eight degrees. Carry an umbrella if you're heading "
     "out before lunch.",
     [("What will the morning be like?", ["Sunny", "Cloudy", "Snowy", "Very hot"], 1, "Cloudy."),
      ("When is rain expected?", ["Morning", "Around noon", "Evening", "Night"], 1, "Around noon."),
      ("What is the advice?", ["Wear sunglasses", "Carry an umbrella before lunch", "Stay home all day",
                               "Wear a jacket at night"], 1, "Carry an umbrella before lunch.")]),
    (3, "A Podcast Clip", "What surprised me most in my research was that productivity isn't really about time "
     "management; it's about energy management. People who scheduled their hardest tasks during their natural peak "
     "hours finished them in roughly half the time, and they reported far less stress.",
     [("What is productivity really about, according to the speaker?", ["Time management", "Energy management",
                                                                        "Money", "Sleep only"], 1,
       "Energy management."),
      ("When should hard tasks be done?", ["Late at night", "During natural peak hours", "After lunch always",
                                           "At the weekend"], 1, "During natural peak hours."),
      ("What was the result?", ["More stress", "Tasks took longer", "Half the time and less stress",
                                "No change"], 2, "Roughly half the time and less stress.")]),
    (3, "Customer Complaint", "I've been charged twice for the same order, and it's been ten days since I raised the "
     "issue. I'd appreciate a refund to my original payment method by the end of this week, otherwise I'll have to "
     "escalate the matter.",
     [("What is the problem?", ["Late delivery", "Charged twice", "Wrong item", "Broken product"], 1,
       "Charged twice."),
      ("What does the customer want?", ["A new product", "A refund", "A discount", "A call"], 1, "A refund."),
      ("What will they do if it isn't solved?", ["Wait more", "Escalate", "Forget it", "Buy again"], 1,
       "Escalate the matter.")]),
]

# -- speaking prompts by module -------------------------------------------------------------
INTERVIEW = [
    (1, "Tell me about yourself."), (1, "Why do you want this job?"), (1, "What are your strengths?"),
    (2, "Describe a challenge you faced at work or college and how you handled it."),
    (2, "Where do you see yourself in five years?"), (2, "Why should we hire you?"),
    (2, "Tell me about a time you worked in a team."), (2, "What is your biggest weakness?"),
    (3, "Describe a time you disagreed with your manager. What did you do?"),
    (3, "How do you prioritise when everything is urgent?"),
    (3, "Tell me about a project you're proud of, and your exact role in it."),
    (3, "What would you do in your first ninety days in this role?"),
]
ROUTINE = [
    (1, "Describe your morning, from waking up to leaving home."), (1, "What do you usually eat in a day?"),
    (1, "Talk about your favourite day of the week and why."), (2, "Describe how you plan your week."),
    (2, "Talk about a habit you want to build and how you'll do it."),
    (2, "Explain how you relax after a busy day."), (3, "Compare your weekday routine with your weekend routine."),
    (3, "Describe a change to your routine that improved your life."),
]
CONVERSATION = [
    (1, "Order food at a restaurant: ask for the menu, order, and ask for the bill."),
    (1, "Introduce yourself to a new neighbour."), (1, "Ask for directions to the railway station."),
    (2, "Call a hotel to book a room and ask about breakfast and check-in time."),
    (2, "Return a faulty product at a shop and ask for a replacement."),
    (2, "Make small talk with a colleague at the coffee machine."),
    (3, "Negotiate a delivery date with a client who wants it earlier."),
    (3, "Politely disagree with a friend's opinion about a movie and explain why."),
]
PRONUNCIATION = [
    (1, "Read aloud: 'I think three thin things are thrilling.'"),
    (1, "Read aloud: 'She sells sea shells by the sea shore.'"),
    (2, "Read aloud: 'Would you like a vegetable sandwich or a very hot vada pav?'"),
    (2, "Read aloud: 'The weather was worse on Wednesday than on Thursday.'"),
    (3, "Read aloud: 'Particularly, the statistics indicate a significantly higher probability.'"),
    (3, "Read aloud: 'I'd rather we thoroughly rethought the strategy, though.'"),
]
