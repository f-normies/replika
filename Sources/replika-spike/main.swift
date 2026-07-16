import Foundation
import ReplikaCore
import SpeechSwiftProvider

// --- Arg parsing (minimal, no external dep) ---
var args = Array(CommandLine.arguments.dropFirst())
var quant: Quant = .q4
var diarize = true
var filePath: String?

while let arg = args.first {
    args.removeFirst()
    switch arg {
    case "--quant":
        guard let v = args.first else {
            FileHandle.standardError.write(Data("error: --quant requires a value (q4|q8)\n".utf8)); exit(2)
        }
        args.removeFirst()
        guard let q = Quant(rawValue: v) else {
            FileHandle.standardError.write(Data("error: invalid --quant '\(v)' (use q4 or q8)\n".utf8)); exit(2)
        }
        quant = q
    case "--no-diarize":
        diarize = false
    default:
        filePath = arg
    }
}

guard let filePath else {
    FileHandle.standardError.write(Data("usage: replika-spike <audio-file> [--quant q4|q8] [--no-diarize]\n".utf8))
    exit(2)
}

let url = URL(fileURLWithPath: filePath)
let clipName = url.lastPathComponent

// Audio duration for RTF denominator.
let samples: [Float]
do {
    samples = try AudioLoader.loadMono16k(url)
} catch {
    FileHandle.standardError.write(Data("error: cannot load audio at \(filePath): \(error)\n".utf8))
    exit(2)
}
let audioSeconds = Double(samples.count) / 16000.0

// Build provider via registry (proves the swap point).
var registry = ProviderRegistry()
registry.register("speech-swift") { SpeechSwiftProvider() }
guard let provider = registry.make("speech-swift") else { fatalError("provider not registered") }

// Peak-memory poller.
let peak = ManagedAtomicPeak()
let poller = Task {
    while !Task.isCancelled {
        peak.update(currentFootprintBytes())
        try? await Task.sleep(nanoseconds: 100_000_000)
    }
}

let clock = ContinuousClock()
let start = clock.now
let options = TranscribeOptions(language: "auto", quant: quant, diarize: diarize)
var segmentCount = 0

for try await ev in provider.transcribe(.file(url), options: options) {
    switch ev {
    case .committed(let seg):
        segmentCount += 1
        let who = seg.speaker.map { "S\($0)" } ?? "S?"
        print("[\(fmt(seg.start))–\(fmt(seg.end))] \(who): \(seg.text)")
    case .done:
        break
    default:
        break
    }
}

let elapsed = clock.now - start
let wall = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
poller.cancel()

let result = BenchResult(
    clip: clipName,
    quant: quant.rawValue,
    rtf: audioSeconds / max(wall, 0.0001),
    peakMB: Double(peak.value) / 1_048_576.0,
    loadAndRunSec: wall,
    segments: segmentCount
)
print("\n" + BenchResult.markdownHeader)
print(result.markdownRow)

func fmt(_ t: Double) -> String { String(format: "%.1f", t) }

final class ManagedAtomicPeak: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: UInt64 = 0
    var value: UInt64 { lock.lock(); defer { lock.unlock() }; return _value }
    func update(_ v: UInt64) { lock.lock(); if v > _value { _value = v }; lock.unlock() }
}
