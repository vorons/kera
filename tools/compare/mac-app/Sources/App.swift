// Kera vs Ollama: one window, one button. It shows each result as it comes in: the scoreboard, two charts, the answer
// being written right now, and every message so far.
import AppKit
import Charts
import SwiftUI

let keraGold = Color(red: 1.0, green: 0.76, blue: 0.10)   // the brand's Bolt gold
let ollamaGrey = Color(red: 0.55, green: 0.58, blue: 0.64)

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let run = TestRun()

    func applicationWillTerminate(_ notification: Notification) {
        run.shutdown()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct KeraVsOllamaApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate

    var body: some Scene {
        WindowGroup("Kera vs Ollama") {
            ContentView()
                .environmentObject(delegate.run)
                .frame(minWidth: 1040, minHeight: 780)
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var run: TestRun

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                progress
                scoreboard
                HStack(alignment: .top, spacing: 16) {
                    panel("Writing speed, each message (tokens a second, higher is better)") { speedChart }
                    panel("Wait for the first token, by kind (seconds, lower is better)") { waitChart }
                }
                HStack(alignment: .top, spacing: 16) {
                    panel(liveTitle) { live }
                    panel("Every message so far") { log }
                }
            }
            .padding(20)
        }
    }

    // MARK: top

    var header: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Kera vs Ollama").font(.largeTitle).bold()
                Text("The same model file, the same messages, one at a time on this Mac. Both answer greedy, with thinking off and capped lengths, to compare speed on equal terms: these aren't the models' usual answers.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
            Picker("Model", selection: $run.modelID) {
                ForEach(modelChoices) { m in
                    Text(m.title).tag(m.id)
                }
            }
            .frame(width: 260)
            .disabled(run.busy)
            Picker("Length", selection: $run.minutes) {
                Text("3 minutes").tag(3)
                Text("10 minutes").tag(10)
                Text("20 minutes").tag(20)
            }
            .frame(width: 170)
            .disabled(run.busy)
            if run.busy {
                Button("Stop") { run.stop() }
                    .controlSize(.large)
            } else {
                Button("Start test") { run.start() }
                    .controlSize(.large)
                    .keyboardShortcut(.defaultAction)
            }
        }
    }

    var progress: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(run.status).font(.headline)
            if let d = run.download {
                ProgressView(value: d)
            } else if run.phase == .running, let s = run.startedAt, let e = run.endsAt {
                TimelineView(.periodic(from: .now, by: 1)) { ctx in
                    let total = e.timeIntervalSince(s)
                    let done = min(total, ctx.date.timeIntervalSince(s))
                    ProgressView(value: done, total: total) {
                        Text(String(format: "%d:%02d of %d:00 (the message under way finishes after the time is up)",
                                    Int(done) / 60, Int(done) % 60, Int(total) / 60))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            } else if run.phase == .preparing {
                ProgressView().progressViewStyle(.linear)
            }
            if let m = modelChoices.first(where: { $0.id == run.modelID }), !run.busy {
                Text(m.detail).font(.caption).foregroundStyle(.secondary)
            }
            if let f = run.resultsFile {
                Button("Show the results file") { NSWorkspace.shared.activateFileViewerSelecting([f]) }
                    .buttonStyle(.link)
            }
        }
    }

    // MARK: scoreboard

    var scoreboard: some View {
        let writing = [Category.chat, .coding, .reasoning]
        return HStack(spacing: 12) {
            card("Writing speed", "tokens a second, chat, code and reasoning",
                 avg(.kera, writing, \.writeTokPerS), avg(.ollama, writing, \.writeTokPerS), higherIsBetter: true, unit: "")
            card("A chat message", "wait for the first token",
                 avg(.kera, [.chat], \.firstWordS), avg(.ollama, [.chat], \.firstWordS), higherIsBetter: false, unit: " s")
            card("A long document", "wait for the first token (reading 16K tokens)",
                 avg(.kera, [.document], \.firstWordS), avg(.ollama, [.document], \.firstWordS), higherIsBetter: false, unit: " s")
            card("Its follow-up", "wait for the first token",
                 avg(.kera, [.followUp], \.firstWordS), avg(.ollama, [.followUp], \.firstWordS), higherIsBetter: false, unit: " s")
            rightCard
        }
    }

    func avg(_ e: Engine, _ cats: [Category], _ key: KeyPath<Outcome, Double>) -> Double? {
        mean(run.results.filter { $0.engine == e && $0.error == nil && cats.contains($0.category) }.map { $0[keyPath: key] })
    }

    func card(_ title: String, _ sub: String, _ k: Double?, _ o: Double?, higherIsBetter: Bool, unit: String) -> some View {
        let fmt = { (x: Double?) -> String in
            guard let x else { return "–" }
            return (x >= 10 ? String(format: "%.0f", x) : String(format: "%.2f", x)) + unit
        }
        var verdict = " "
        if let k, let o, k > 0, o > 0 {
            let r = higherIsBetter ? k / o : o / k
            let word = higherIsBetter ? "as fast" : "sooner"
            verdict = r >= 1 ? String(format: "Kera %.2f× %@", r, word) : String(format: "Ollama %.2f× %@", 1 / r, word)
        }
        return VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.headline)
            Text(sub).font(.caption).foregroundStyle(.secondary).lineLimit(2, reservesSpace: true)
            HStack(alignment: .firstTextBaseline) {
                Text(fmt(k)).font(.title2.monospacedDigit()).bold().foregroundStyle(keraGold)
                Text("Kera").font(.caption)
            }
            HStack(alignment: .firstTextBaseline) {
                Text(fmt(o)).font(.title2.monospacedDigit()).foregroundStyle(ollamaGrey)
                Text("Ollama").font(.caption)
            }
            Text(verdict).font(.callout.bold())
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
    }

    var rightCard: some View {
        let checked = run.results.filter { $0.correct != nil }
        let line = { (e: Engine) -> String in
            let mine = checked.filter { $0.engine == e }
            return mine.isEmpty ? "–" : "\(mine.filter { $0.correct == true }.count) of \(mine.count)"
        }
        return VStack(alignment: .leading, spacing: 6) {
            Text("Right answers").font(.headline)
            Text("sums and questions about the document (greedy, thinking off)").font(.caption).foregroundStyle(.secondary)
                .lineLimit(2, reservesSpace: true)
            HStack(alignment: .firstTextBaseline) {
                Text(line(.kera)).font(.title2.monospacedDigit()).bold().foregroundStyle(keraGold)
                Text("Kera").font(.caption)
            }
            HStack(alignment: .firstTextBaseline) {
                Text(line(.ollama)).font(.title2.monospacedDigit()).foregroundStyle(ollamaGrey)
                Text("Ollama").font(.caption)
            }
            Text(" ").font(.callout)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
    }

    // MARK: charts

    var speedChart: some View {
        let shown = run.results.filter { $0.error == nil && $0.writeTokPerS > 0 && [.chat, .coding, .reasoning].contains($0.category) }
        return Chart(shown) { r in
            LineMark(x: .value("Message", r.pair), y: .value("Tokens a second", r.writeTokPerS),
                     series: .value("Engine", r.engine.rawValue))
                .foregroundStyle(by: .value("Engine", r.engine.rawValue))
            PointMark(x: .value("Message", r.pair), y: .value("Tokens a second", r.writeTokPerS))
                .foregroundStyle(by: .value("Engine", r.engine.rawValue))
        }
        .chartForegroundStyleScale(["Kera": keraGold, "Ollama": ollamaGrey])
        .chartXAxisLabel("message")
        .frame(height: 240)
    }

    struct Bar: Identifiable {
        let id = UUID()
        let kind: String
        let engine: String
        let value: Double
    }

    var waitChart: some View {
        var bars: [Bar] = []
        for c in Category.allCases {
            for e in Engine.allCases {
                if let v = avg(e, [c], \.firstWordS) {
                    bars.append(Bar(kind: c.rawValue, engine: e.rawValue, value: v))
                }
            }
        }
        return Chart(bars) { b in
            BarMark(x: .value("Kind", b.kind), y: .value("Seconds", b.value))
                .foregroundStyle(by: .value("Engine", b.engine))
                .position(by: .value("Engine", b.engine))
                .annotation(position: .top) {
                    Text(b.value >= 10 ? String(format: "%.0f", b.value) : String(format: "%.1f", b.value))
                        .font(.caption2)
                }
        }
        .chartForegroundStyleScale(["Kera": keraGold, "Ollama": ollamaGrey])
        .frame(height: 240)
    }

    // MARK: live

    var liveTitle: String {
        guard let e = run.liveEngine, let c = run.liveCategory else { return "Now" }
        let rate = run.liveRate > 0 ? String(format: ", %.0f tokens a second", run.liveRate) : ""
        return "Now: \(c.rawValue) on \(e.rawValue)\(rate)"
    }

    var live: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Text(run.liveText.isEmpty ? (run.busy ? "(reading the prompt...)" : "") : String(run.liveText.suffix(3000)))
                    .font(.system(.callout, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                Color.clear.frame(height: 1).id("end")
            }
            .onChange(of: run.liveText) { proxy.scrollTo("end") }
        }
        .frame(height: 260)
    }

    var log: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 4) {
                ForEach(Array(run.results.reversed())) { r in
                    HStack(spacing: 8) {
                        Circle().fill(r.engine == .kera ? keraGold : ollamaGrey).frame(width: 8, height: 8)
                        Text("#\(r.pair) \(r.category.rawValue)").frame(width: 150, alignment: .leading)
                        Text(r.engine.rawValue).frame(width: 55, alignment: .leading)
                        if let e = r.error {
                            Text(e).foregroundStyle(.red).lineLimit(1)
                        } else {
                            Text(String(format: "first token %.2f s", r.firstWordS)).frame(width: 130, alignment: .leading)
                            Text(String(format: "%.0f tok/s", r.writeTokPerS)).frame(width: 80, alignment: .leading)
                            Text(String(format: "%.1f s in all", r.totalS)).frame(width: 95, alignment: .leading)
                            if let ok = r.correct {
                                Text(ok ? "right" : "wrong").foregroundStyle(ok ? .green : .red)
                            }
                        }
                    }
                    .font(.caption.monospacedDigit())
                }
            }
        }
        .frame(height: 260)
    }

    func panel<V: View>(_ title: String, @ViewBuilder _ content: () -> V) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.primary.opacity(0.05)))
    }
}
