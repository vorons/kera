// The test itself: gets Ollama and the model ready, starts Kera on Ollama's copy of the model, then asks both the same
// messages in turn until the time is up, keeping every result as it comes.
import Darwin
import Foundation
import SwiftUI

struct Outcome: Identifiable, Codable {
    var id = UUID()
    let pair: Int            // the message's number: the same for both engines
    let round: Int
    let category: Category
    let engine: Engine
    let firstWordS: Double   // from sending it to the first token
    let totalS: Double       // from sending it to the last token
    let written: Int         // tokens written (the engine's own count)
    let writeTokPerS: Double // writing speed (the engine's own count: tokens after the first, over their time)
    let readTokens: Int      // prompt tokens read (not already held)
    let correct: Bool?       // nil where there is nothing to check
    let error: String?
}

struct Answer {
    var text = ""
    var firstWordS: Double?
    var totalS = 0.0
    var written = 0
    var writeTokPerS = 0.0
    var readTokens = 0
}

enum Phase: Equatable {
    case idle, preparing, running, done, failed
}

struct TestError: LocalizedError {
    let message: String
    init(_ m: String) { message = m }
    var errorDescription: String? { message }
}

@MainActor
final class TestRun: ObservableObject {
    @Published var modelID = modelChoices[0].id
    @Published var minutes = 10
    @Published var phase = Phase.idle
    @Published var status = "Press Start test. The first time, the model is downloaded first (about 20 GB)."
    @Published var download: Double?             // 0...1 while downloading
    @Published var results: [Outcome] = []
    @Published var liveEngine: Engine?
    @Published var liveCategory: Category?
    @Published var liveText = ""
    @Published var liveRate = 0.0
    @Published var startedAt: Date?
    @Published var endsAt: Date?
    @Published var resultsFile: URL?

    let folder: URL
    private var task: Task<Void, Never>?
    private var keraProcess: Process?
    private var ollamaProcess: Process?
    private var keraPort = 0
    private var ollamaBase = ""
    private let session: URLSession

    init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        folder = base.appendingPathComponent("KeraVsOllama", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let c = URLSessionConfiguration.default
        c.timeoutIntervalForRequest = 1800   // reading a long document sends nothing for a while
        c.timeoutIntervalForResource = 7 * 24 * 3600
        session = URLSession(configuration: c)
    }

    var busy: Bool { phase == .preparing || phase == .running }

    func start() {
        guard !busy else { return }
        stopServers() // a run before this one leaves no server behind (a model loaded twice would skew the race)
        results = []
        liveText = ""
        resultsFile = nil
        task = Task { await self.go() }
    }

    func stop() {
        task?.cancel()
        status = "Stopping..."
    }

    // when the app quits: the servers this app started stop with it
    func shutdown() {
        task?.cancel()
        stopServers()
    }

    private func stopServers() {
        for p in [keraProcess, ollamaProcess].compactMap({ $0 }) where p.isRunning {
            p.interrupt()
            for _ in 0..<50 where p.isRunning {
                usleep(100_000)
            }
            if p.isRunning {
                p.terminate()
            }
        }
        keraProcess = nil
        ollamaProcess = nil
    }

    // MARK: the run

    private func go() async {
        phase = .preparing
        do {
            try await prepareOllama()
            try await pullModel()
            let file = try await modelFile()
            try await startKera(file)
            status = "Warming up both (each loads the model into memory once)..."
            _ = try await ask(.kera, [["role": "user", "content": "Say hello."]], 8)
            _ = try await ask(.ollama, [["role": "user", "content": "Say hello."]], 8)
            try Task.checkCancellation()

            phase = .running
            let begin = Date()
            startedAt = begin
            endsAt = begin.addingTimeInterval(Double(minutes) * 60)
            var round = 0
            var pair = 0
            outer: while !Task.isCancelled {
                let doc = makeDocument(round: round)
                // who goes first changes each round, so neither always has the cooler chip
                let order: [Engine] = round % 2 == 0 ? [.kera, .ollama] : [.ollama, .kera]
                var firstAnswers: [Engine: String] = [:]
                for job in roundJobs(round, doc) {
                    if Date() >= endsAt! || Task.isCancelled { break outer }
                    pair += 1
                    for e in order {
                        firstAnswers[e] = await run(job, on: e, pair: pair, round: round)
                    }
                }
                if Date() >= endsAt! || Task.isCancelled { break }
                pair += 1
                for e in order {
                    _ = await run(followUpJob(doc, firstAnswer: firstAnswers[e] ?? ""), on: e, pair: pair, round: round)
                }
                round += 1
            }
            liveEngine = nil
            liveCategory = nil
            phase = .done
            status = Task.isCancelled ? "Stopped. " + summary() : "Done. " + summary()
        } catch {
            if Task.isCancelled {
                phase = results.isEmpty ? .idle : .done
                status = "Stopped. " + summary()
            } else {
                phase = .failed
                status = "Couldn't finish: \(error.localizedDescription) (logs in \(folder.path))"
            }
        }
        liveEngine = nil
        liveCategory = nil
        save()
    }

    private func run(_ job: Job, on e: Engine, pair: Int, round: Int) async -> String {
        liveEngine = e
        liveCategory = job.category
        status = "\(job.category.rawValue) on \(e.rawValue): \(job.category.what)"
        do {
            let a = try await ask(e, job.messages, job.maxTokens)
            results.append(Outcome(pair: pair, round: round, category: job.category, engine: e,
                                  firstWordS: a.firstWordS ?? a.totalS, totalS: a.totalS, written: a.written,
                                  writeTokPerS: a.writeTokPerS, readTokens: a.readTokens,
                                  correct: job.check.map { $0(a.text) }, error: nil))
            save()
            return a.text
        } catch {
            if Task.isCancelled { return "" }
            results.append(Outcome(pair: pair, round: round, category: job.category, engine: e, firstWordS: 0,
                                  totalS: 0, written: 0, writeTokPerS: 0, readTokens: 0, correct: nil,
                                  error: error.localizedDescription))
            return ""
        }
    }

    func summary() -> String {
        let k = averageWriting(.kera)
        let o = averageWriting(.ollama)
        guard let k, let o, o > 0 else { return "" }
        return String(format: "Kera wrote %.0f tokens a second, Ollama %.0f: %.2f× as fast.", k, o, k / o)
    }

    func averageWriting(_ e: Engine) -> Double? {
        mean(results.filter { $0.engine == e && $0.error == nil && [.chat, .coding, .reasoning].contains($0.category) }
            .map(\.writeTokPerS))
    }

    private func save() {
        guard !results.isEmpty else { return }
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd-HHmm"
        let url = resultsFile ?? folder.appendingPathComponent("results-\(f.string(from: startedAt ?? Date())).json")
        struct Saved: Codable {
            let model: String
            let minutes: Int
            let started: Date?
            let results: [Outcome]
        }
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        enc.dateEncodingStrategy = .iso8601
        if let data = try? enc.encode(Saved(model: modelID, minutes: minutes, started: startedAt, results: results)) {
            try? data.write(to: url)
            resultsFile = url
        }
    }

    // MARK: Ollama

    // The Ollama already running on this Mac if there is one (the one you use), else Ollama's own program started on a
    // private port: the installed one, or downloaded into this app's folder
    private func prepareOllama() async throws {
        status = "Looking for Ollama..."
        if await answers("http://127.0.0.1:11434/api/version") {
            ollamaBase = "http://127.0.0.1:11434"
            return
        }
        var exe = ["/Applications/Ollama.app/Contents/Resources/ollama", "/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
        if exe == nil {
            exe = try await downloadOllama()
        }
        guard let exe else { throw TestError("Ollama couldn't be found or downloaded") }
        ollamaBase = "http://127.0.0.1:11500"
        status = "Starting Ollama..."
        ollamaProcess = try launch(URL(fileURLWithPath: exe), ["serve"], env: ["OLLAMA_HOST": "127.0.0.1:11500"],
                                   log: folder.appendingPathComponent("ollama-serve.log"))
        for _ in 0..<120 {
            if await answers(ollamaBase + "/api/version") { return }
            if ollamaProcess?.isRunning != true { break }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        throw TestError("Ollama didn't start (its log: ollama-serve.log)")
    }

    private func downloadOllama() async throws -> String? {
        let dir = folder.appendingPathComponent("ollama", isDirectory: true)
        if let found = findExecutable(named: "ollama", in: dir) { return found }
        status = "Downloading Ollama (it isn't on this Mac)..."
        let url = URL(string: "https://github.com/ollama/ollama/releases/latest/download/ollama-darwin.tgz")!
        let (file, response) = try await session.download(from: url)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw TestError("Ollama's download failed") }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let code = try await runAndWait(URL(fileURLWithPath: "/usr/bin/tar"), ["-xzf", file.path, "-C", dir.path])
        guard code == 0 else { throw TestError("Ollama's download couldn't be unpacked") }
        return findExecutable(named: "ollama", in: dir)
    }

    private func pullModel() async throws {
        status = "Getting the model through Ollama (\(modelID))..."
        var req = URLRequest(url: URL(string: ollamaBase + "/api/pull")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: ["model": modelID, "stream": true])
        let (bytes, response) = try await session.bytes(for: req)
        try await expectOK(response, bytes)
        for try await line in bytes.lines {
            guard let j = json(line) else { continue }
            if let e = j["error"] as? String { throw TestError("Ollama couldn't download the model: \(e)") }
            let total = num(j["total"])
            let done = num(j["completed"])
            if total > 1e9 {
                download = done / total
                status = String(format: "Downloading the model: %.1f of %.1f GB", done / 1e9, total / 1e9)
            }
            if (j["status"] as? String) == "success" { break }
        }
        download = nil
    }

    // the model's file in Ollama's folder, which Kera runs too: both engines run the very same weights
    private func modelFile() async throws -> String {
        var req = URLRequest(url: URL(string: ollamaBase + "/api/show")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: ["model": modelID])
        let (data, response) = try await session.data(for: req)
        guard (response as? HTTPURLResponse)?.statusCode == 200,
              let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelfile = j["modelfile"] as? String else {
            throw TestError("Ollama didn't say where the model's file is")
        }
        for line in modelfile.split(separator: "\n") where line.hasPrefix("FROM /") {
            let path = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)
            if FileManager.default.fileExists(atPath: path) { return path }
        }
        throw TestError("Ollama's copy of the model couldn't be found")
    }

    // MARK: Kera

    private func startKera(_ file: String) async throws {
        guard let exe = Bundle.main.url(forResource: "kera", withExtension: nil) else {
            throw TestError("this app's copy of Kera is missing")
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        if !FileManager.default.fileExists(atPath: home.appendingPathComponent("Library/Application Support/Kera/kera_machine.json").path) {
            status = "Kera is measuring this Mac (a few seconds)..."
            _ = try await runAndWait(exe, ["setup", "--quick"])
        }
        keraPort = (18080...18199).first(where: portFree) ?? 18080
        status = "Starting Kera on the same model file..."
        keraProcess = try launch(exe, ["serve", "-m", file, "-c", String(contextTokens), "-np", "1", "--host", "127.0.0.1",
                                       "--port", String(keraPort)],
                                 env: [:], log: folder.appendingPathComponent("kera-serve.log"))
        for _ in 0..<1800 {
            if await answers("http://127.0.0.1:\(keraPort)/health") { return }
            if keraProcess?.isRunning != true { break }
            try await Task.sleep(nanoseconds: 500_000_000)
        }
        throw TestError("Kera didn't start (its log: kera-serve.log)")
    }

    // MARK: asking

    private func ask(_ e: Engine, _ messages: [Message], _ maxTokens: Int) async throws -> Answer {
        liveText = ""
        liveRate = 0
        if e == .kera {
            return try await askKera(messages, maxTokens)
        }
        return try await askOllama(messages, maxTokens)
    }

    private func askKera(_ messages: [Message], _ maxTokens: Int) async throws -> Answer {
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(keraPort)/v1/chat/completions")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["messages": messages, "stream": true, "max_tokens": maxTokens, "temperature": 0,
                                   "top_k": 1, "chat_template_kwargs": ["enable_thinking": false], "cache_prompt": true]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let t0 = Date()
        let (bytes, response) = try await session.bytes(for: req)
        try await expectOK(response, bytes)
        var a = Answer()
        var pieces = 0
        for try await line in bytes.lines {
            try Task.checkCancellation()
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))
            if payload == "[DONE]" { break }
            guard let j = json(payload) else { continue }
            if let t = j["timings"] as? [String: Any] {
                a.written = Int(num(t["predicted_n"]))
                a.writeTokPerS = num(t["predicted_per_second"])
                a.readTokens = Int(num(t["prompt_n"]))
            }
            for c in (j["choices"] as? [[String: Any]]) ?? [] {
                let d = c["delta"] as? [String: Any] ?? [:]
                let piece = (d["content"] as? String ?? "") + (d["reasoning_content"] as? String ?? "")
                if !piece.isEmpty {
                    note(piece, &a, &pieces, t0)
                }
            }
        }
        a.totalS = Date().timeIntervalSince(t0)
        return a
    }

    private func askOllama(_ messages: [Message], _ maxTokens: Int) async throws -> Answer {
        var req = URLRequest(url: URL(string: ollamaBase + "/api/chat")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = ["model": modelID, "messages": messages, "stream": true, "think": false,
                                   "keep_alive": "30m",
                                   "options": ["temperature": 0, "top_k": 1, "num_ctx": contextTokens,
                                               "num_predict": maxTokens, "seed": 1]]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let t0 = Date()
        let (bytes, response) = try await session.bytes(for: req)
        try await expectOK(response, bytes)
        var a = Answer()
        var pieces = 0
        for try await line in bytes.lines {
            try Task.checkCancellation()
            guard let j = json(line) else { continue }
            if let e = j["error"] as? String { throw TestError("Ollama: \(e)") }
            let m = j["message"] as? [String: Any] ?? [:]
            let piece = (m["content"] as? String ?? "") + (m["thinking"] as? String ?? "")
            if !piece.isEmpty {
                note(piece, &a, &pieces, t0)
            }
            if (j["done"] as? Bool) == true {
                a.written = Int(num(j["eval_count"]))
                let ns = num(j["eval_duration"])
                a.writeTokPerS = ns > 0 ? num(j["eval_count"]) / (ns / 1e9) : 0
                a.readTokens = Int(num(j["prompt_eval_count"]))
                break
            }
        }
        a.totalS = Date().timeIntervalSince(t0)
        return a
    }

    // a streamed piece: the live view, and the time of the first one
    private func note(_ piece: String, _ a: inout Answer, _ pieces: inout Int, _ t0: Date) {
        let now = Date().timeIntervalSince(t0)
        if a.firstWordS == nil {
            a.firstWordS = now
        }
        a.text += piece
        pieces += 1
        liveText = a.text
        if let first = a.firstWordS, now > first + 0.2 {
            liveRate = Double(pieces - 1) / (now - first)
        }
    }

    // MARK: helpers

    private func expectOK(_ response: URLResponse, _ bytes: URLSession.AsyncBytes) async throws {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code != 200 else { return }
        var text = ""
        for try await line in bytes.lines {
            text += line
            if text.count > 400 { break }
        }
        throw TestError("HTTP \(code): \(text)")
    }

    private func answers(_ url: String) async -> Bool {
        var req = URLRequest(url: URL(string: url)!)
        req.timeoutInterval = 2
        guard let r = try? await session.data(for: req) else { return false }
        return (r.1 as? HTTPURLResponse)?.statusCode == 200
    }
}

func mean(_ xs: [Double]) -> Double? {
    xs.isEmpty ? nil : xs.reduce(0, +) / Double(xs.count)
}

func num(_ x: Any?) -> Double {
    (x as? NSNumber)?.doubleValue ?? 0
}

func json(_ line: String) -> [String: Any]? {
    guard let data = line.data(using: .utf8) else { return nil }
    return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
}

func launch(_ exe: URL, _ args: [String], env: [String: String], log: URL) throws -> Process {
    let p = Process()
    p.executableURL = exe
    p.arguments = args
    var e = ProcessInfo.processInfo.environment
    for (k, v) in env {
        e[k] = v
    }
    p.environment = e
    FileManager.default.createFile(atPath: log.path, contents: nil)
    let h = try FileHandle(forWritingTo: log)
    p.standardOutput = h
    p.standardError = h
    try p.run()
    return p
}

func runAndWait(_ exe: URL, _ args: [String]) async throws -> Int32 {
    try await withCheckedThrowingContinuation { cont in
        let p = Process()
        p.executableURL = exe
        p.arguments = args
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        p.terminationHandler = { cont.resume(returning: $0.terminationStatus) }
        do {
            try p.run()
        } catch {
            cont.resume(throwing: error)
        }
    }
}

func findExecutable(named name: String, in dir: URL) -> String? {
    guard let items = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil) else { return nil }
    for case let url as URL in items where url.lastPathComponent == name {
        if FileManager.default.isExecutableFile(atPath: url.path) { return url.path }
    }
    return nil
}

func portFree(_ port: Int) -> Bool {
    let s = socket(AF_INET, SOCK_STREAM, 0)
    if s < 0 { return false }
    defer { close(s) }
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(UInt16(port).bigEndian)
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    let r = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(s, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    return r == 0
}
