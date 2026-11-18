// What the test asks both engines: the same messages, in the same order, greedy, with thinking off. No message is
// asked twice in a test (see roundJobs).
import Foundation

enum Engine: String, CaseIterable, Identifiable, Codable {
    case kera = "Kera"
    case ollama = "Ollama"
    var id: String { rawValue }
}

enum Category: String, CaseIterable, Identifiable, Codable {
    case chat = "Chat"
    case coding = "Coding"
    case reasoning = "Reasoning"
    case document = "Long document"
    case followUp = "Follow-up"
    var id: String { rawValue }

    // what it measures, in a few words
    var what: String {
        switch self {
        case .chat: return "a short question, up to 200 tokens"
        case .coding: return "writing code, up to 400 tokens"
        case .reasoning: return "a sum worked out in words, the answer checked"
        case .document: return "a 16K-token log, then a question about it, the answer checked"
        case .followUp: return "a second question on the same log, the answer checked"
        }
    }
}

struct ModelChoice: Identifiable, Hashable {
    let id: String      // Ollama's name for it
    let title: String
    let detail: String
}

// Mixture-of-experts models both engines run from the same file (Ollama downloads it; Kera reads Ollama's copy)
let modelChoices: [ModelChoice] = [
    ModelChoice(id: "nemotron-3-nano:30b", title: "Nemotron 3 Nano 30B",
                detail: "NVIDIA's mixture of experts: 30B in all, about 3B for each token (about 24 GB to download)"),
    ModelChoice(id: "qwen3:30b", title: "Qwen3 30B-A3B",
                detail: "Alibaba's mixture of experts: 30B in all, about 3B for each token (about 19 GB to download)"),
]

let contextTokens = 32768   // the same chat memory on both
let documentTokens = 16000  // about this long

typealias Message = [String: String]

struct Job {
    let category: Category
    let messages: [Message]
    let maxTokens: Int
    let check: ((String) -> Bool)?
}

// A seeded random source, so both engines get the same document in each round
struct SplitMix {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func int(_ lo: Int, _ hi: Int) -> Int { lo + Int(next() % UInt64(hi - lo + 1)) }
    mutating func pick<T>(_ xs: [T]) -> T { xs[Int(next() % UInt64(xs.count))] }
}

struct Document {
    let text: String
    let question: String
    let answer: String
    let followUp: String
    let followUpAnswer: String
}

// A shipping log about documentTokens long, with two notes hidden in it: a vault's code a third of the way in, and
// where a warehouse moved two thirds of the way in
func makeDocument(round: Int) -> Document {
    var r = SplitMix(state: 0x5EED_0000 &+ UInt64(round) &* 7919)
    let goods = ["copper wire", "olive oil", "bicycle parts", "printer paper", "ceramic tiles", "coffee beans",
                 "wool blankets", "glass jars", "solar panels", "rice", "steel bolts", "tea"]
    let days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    let cities = ["Lyon", "Porto", "Gdansk", "Leeds", "Turin", "Ghent", "Bilbao", "Malmo", "Graz", "Cork"]
    let teams = ["Falcon", "Heron", "Otter", "Lynx", "Badger", "Raven", "Marten", "Kestrel"]
    let far = ["Reykjavik", "Valparaiso", "Tbilisi", "Windhoek", "Hobart", "Ushuaia", "Nuuk", "Ulaanbaatar"]
    let lines = documentTokens / 24
    let vault = r.int(1000, 9999)
    let code = String(r.int(100_000, 999_999))
    let warehouse = r.int(100, 999)
    let city = r.pick(far)
    var text = "Shipping log, season \(round + 1). Each line is one shipment.\n\n"
    for i in 0..<lines {
        if i == lines / 3 {
            text += "Note: the access code for vault \(vault) is \(code).\n"
        }
        if i == lines * 2 / 3 {
            text += "Note: warehouse W-\(warehouse) moved to \(city).\n"
        }
        text += "Record \(10000 + i): \(r.int(2, 90)) crates of \(r.pick(goods)) left Dock \(r.int(1, 12)) on "
            + "\(r.pick(days)) for \(r.pick(cities)), checked by team \(r.pick(teams)).\n"
    }
    return Document(text: text,
                    question: "What is the access code for vault \(vault)? Reply with the code only.",
                    answer: code,
                    followUp: "Which city did warehouse W-\(warehouse) move to? Reply with the city only.",
                    followUpAnswer: city)
}

let chatPrompts = [
    "Explain how a refrigerator keeps food cold.", "Describe the history of the printing press.",
    "How do vaccines teach the body to fight a disease?", "Write a short story about a lighthouse keeper.",
    "What makes a good night's sleep, and why does it matter?", "Explain how bridges carry heavy loads.",
    "Describe a day in the life of a honeybee.", "Explain why the sky is blue and sunsets are red.",
    "Explain how a bicycle stays upright while it moves.", "Describe how bread dough rises, and why it needs time.",
    "Write a short story about a girl who finds a map inside an old book.",
    "How does a city bring clean water to every tap?",
]

let codingPrompts = [
    "Write a Python function that merges overlapping intervals in a list of (start, end) pairs, with a docstring and three small tests.",
    "Write a JavaScript function that debounces another function, with a short usage example.",
    "Write a Swift struct for a to-do item and a function that sorts a list of them by due date, then by title.",
    "Write a SQL query that finds each customer's three most recent orders, given tables customers(id, name) and orders(id, customer_id, placed_at, total).",
    "Write a Rust function that counts the words in a string and returns the ten most common with their counts.",
    "Write a Python class for a least-recently-used cache with get and put in constant time, with a short test.",
    "Write a Go function that checks whether a string of (), [] and {} brackets is balanced, with three small tests.",
    "Write a TypeScript function that groups a list of objects by the value of one of their keys, with a short usage example.",
    "Write a C function that reverses a singly linked list in place, with a small main that builds a list and prints it reversed.",
    "Write a Kotlin data class for a book and a function that returns the titles of books published after a given year, sorted by author.",
    "Write a Bash script that lists the ten largest files under a folder, with their sizes in megabytes.",
    "Write a Java method that returns the n-th Fibonacci number without recursion, with a short test in main.",
]

// (the question, its answer)
let reasoningPrompts: [(String, String)] = [
    ("A train leaves at 9:40 and arrives at 13:05 the same day. How many minutes does the trip take?", "205"),
    ("Sara has three times as many apples as Tom. Together they have 48 apples. How many apples does Sara have?", "36"),
    ("A shop sells pens at 3 for 2 dollars. How many dollars do 27 pens cost?", "18"),
    ("A rectangle is 12 cm by 7 cm. A 3 cm by 3 cm square is cut from one corner. What area is left, in square centimetres?", "75"),
    ("What is 17 times 24, minus 19?", "389"),
    ("A tank holds 240 litres and is a quarter full. How many litres must be added to make it three quarters full?", "120"),
    ("Five friends share a bill of 135 dollars equally, then each adds a 3 dollar tip. How many dollars does each pay in all?", "30"),
    ("A cyclist rides at 18 km an hour for 2 hours and 20 minutes. How many kilometres does she ride?", "42"),
    ("A baker makes 14 trays of 18 rolls and sells all but 25 of them. How many rolls does she sell?", "227"),
    ("A car uses 6 litres of fuel for every 100 km. How many litres does it use on a 350 km trip?", "21"),
    ("A book has 312 pages. Tom reads 24 pages a day. How many days does he take to read it all?", "13"),
    ("A school has 18 classes of 27 pupils, and 39 pupils are away today. How many pupils are in school today?", "447"),
]

// the first number after the last "answer" in the text
func answerNumber(_ text: String) -> String {
    let lower = text.lowercased()
    guard let r = lower.range(of: "answer", options: .backwards) else { return "" }
    var digits = ""
    for ch in lower[r.upperBound...] {
        if ch.isNumber {
            digits.append(ch)
        } else if !digits.isEmpty && ch != "," {
            break
        }
    }
    return digits
}

func contains(_ text: String, _ word: String) -> Bool {
    text.range(of: word, options: .caseInsensitive) != nil
}

// One round: a chat, some code, a sum, and the document's first question (its follow-up is asked after each engine's
// own answer to it). No message is asked twice in a test: Kera guesses ahead by copying from the chat (llama.cpp's
// ngram-mod, one table for every request), so a message asked again would let it copy its own earlier answer and
// look faster than it is. Once a kind's messages are used up, later rounds leave that kind out; each round's
// document is new.
func roundJobs(_ round: Int, _ doc: Document) -> [Job] {
    var jobs: [Job] = []
    if round < chatPrompts.count {
        jobs.append(Job(category: .chat, messages: [["role": "user", "content": chatPrompts[round]]],
                        maxTokens: 200, check: nil))
    }
    if round < codingPrompts.count {
        jobs.append(Job(category: .coding, messages: [["role": "user", "content": codingPrompts[round]]],
                        maxTokens: 400, check: nil))
    }
    if round < reasoningPrompts.count {
        let (sum, result) = reasoningPrompts[round]
        jobs.append(Job(category: .reasoning,
                        messages: [["role": "user", "content": sum + " Work it out in a few short steps, then end with 'Answer: <number>'."]],
                        maxTokens: 400, check: { answerNumber($0) == result }))
    }
    jobs.append(Job(category: .document, messages: documentMessages(doc), maxTokens: 60,
                    check: { contains($0, doc.answer) }))
    return jobs
}

func documentMessages(_ doc: Document) -> [Message] {
    [["role": "user", "content": doc.text + "\n\n" + doc.question]]
}

func followUpJob(_ doc: Document, firstAnswer: String) -> Job {
    Job(category: .followUp,
        messages: documentMessages(doc) + [["role": "assistant", "content": firstAnswer],
                                           ["role": "user", "content": doc.followUp]],
        maxTokens: 60, check: { contains($0, doc.followUpAnswer) })
}
