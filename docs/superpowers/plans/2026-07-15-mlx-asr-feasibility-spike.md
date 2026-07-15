# MLX-ASR Feasibility Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that file-based Qwen3-ASR transcription with word-level timestamps and speaker diarization runs natively on Apple Silicon via the `speech-swift` package, behind our own `TranscriptionProvider` protocol, with measured speed/memory.

**Architecture:** A dependency-free `ReplikaCore` SPM target holds the domain types, the `TranscriptionProvider` protocol, audio decoding and the pure merge/segment logic (all TDD-tested, no MLX). A separate `SpeechSwiftProvider` target wraps `speech-swift` (Qwen3-ASR + ForcedAligner + Sortformer diarization) behind that protocol. A CLI (`replika-spike`) drives it and benchmarks RTF / peak memory. A throwaway `asr-probe` executable reconciles the exact `speech-swift` API against the compiler first, before the real wrapper is written.

**Tech Stack:** Swift 6, SwiftPM, AVFoundation, `os.Logger`, swift-testing, MLX via `soniqo/speech-swift` (Qwen3ASR + SpeechVAD targets).

## Global Constraints

- Minimum platform: **macOS 15** (`.macOS(.v15)`) — required by `speech-swift`'s `MLState`.
- Toolchain: **Swift 6** (`// swift-tools-version: 6.0`), Xcode 16+ with Metal Toolchain, Apple Silicon only.
- Dependency: `.package(url: "https://github.com/soniqo/speech-swift", branch: "main")` — products `Qwen3ASR` and `SpeechVAD`.
- Models auto-download from HuggingFace on first `fromPretrained()` into `~/Library/Caches/qwen3-speech/`. Model-cache relocation is **out of scope** (later slice).
- Style: files 200–400 lines; `struct`/immutability; `os.Logger` (never `print` in library code); factory/registry for providers; swift-testing for tests.
- Any code calling `speech-swift` (Qwen3ASRModel, Qwen3ForcedAligner, DiarizationPipeline) must be reconciled against the compiler — exact signatures/property names come from Task 2's probe, which is authoritative if they differ from the snippets below.
- `speech-swift` API as documented (reconcile at Task 2): `Qwen3ASRModel.fromPretrained()` async throws; `.transcribe(audio: [Float], sampleRate: Int) -> String` sync; `Qwen3ForcedAligner.fromPretrained()` async throws; `.align(audio:text:sampleRate:)` → words with `.text/.startTime/.endTime`; `DiarizationPipeline.fromPretrained()` (import `SpeechVAD`) async throws; `.diarize(audio:sampleRate:)` → segments with `.speakerId/.startTime/.endTime`.

---

### Task 1: SwiftPM skeleton + speech-swift dependency builds

Proves the toolchain resolves and compiles `speech-swift` on this machine — the biggest "does it even build" risk — before any real code.

**Files:**
- Create: `Package.swift`
- Create: `Sources/ReplikaCore/Placeholder.swift`
- Create: `Sources/SpeechSwiftProvider/Placeholder.swift`
- Create: `Sources/asr-probe/main.swift`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: a buildable package with targets `ReplikaCore`, `SpeechSwiftProvider`, `asr-probe`, and the `speech-swift` dependency wired to `Qwen3ASR` + `SpeechVAD`.

- [ ] **Step 1: Write `.gitignore`**

```
.build/
.swiftpm/
*.xcodeproj
.DS_Store
```

- [ ] **Step 2: Write `Package.swift`**

```swift
// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Replika",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/soniqo/speech-swift", branch: "main")
    ],
    targets: [
        .target(name: "ReplikaCore"),
        .target(
            name: "SpeechSwiftProvider",
            dependencies: [
                "ReplikaCore",
                .product(name: "Qwen3ASR", package: "speech-swift"),
                .product(name: "SpeechVAD", package: "speech-swift")
            ]
        ),
        .executableTarget(
            name: "asr-probe",
            dependencies: [.product(name: "Qwen3ASR", package: "speech-swift")]
        )
    ]
)
```

- [ ] **Step 3: Write placeholder sources so the library targets are non-empty**

`Sources/ReplikaCore/Placeholder.swift`:
```swift
// Placeholder — replaced in Task 3.
enum ReplikaCorePlaceholder {}
```

`Sources/SpeechSwiftProvider/Placeholder.swift`:
```swift
// Placeholder — replaced in Task 7.
enum SpeechSwiftProviderPlaceholder {}
```

`Sources/asr-probe/main.swift`:
```swift
print("asr-probe: build ok")
```

- [ ] **Step 4: Resolve and build**

Run: `swift build`
Expected: dependency `speech-swift` resolves and the package compiles. If it fails on toolchain/Metal, STOP and record the exact error — that is itself a spike finding.

- [ ] **Step 5: Run the probe executable**

Run: `swift run asr-probe`
Expected: prints `asr-probe: build ok`

- [ ] **Step 6: Commit**

```bash
git add Package.swift .gitignore Sources/
git commit -m "chore: SwiftPM skeleton with speech-swift dependency"
```

---

### Task 2: ASR probe — reconcile the real speech-swift API end-to-end

Loads Qwen3-ASR, transcribes one real clip, prints text. This is the authoritative reconciliation of `speech-swift`'s exact signatures (async? throwing? property names?). Downstream tasks use whatever this task confirms.

**Files:**
- Modify: `Sources/asr-probe/main.swift`

**Interfaces:**
- Consumes: `Qwen3ASR` product from Task 1.
- Produces: confirmed call shapes for `Qwen3ASRModel.fromPretrained()`, `.transcribe`, `Qwen3ForcedAligner.fromPretrained()`, `.align` (written into a comment block at the top of `main.swift` as the reconciled reference).

- [ ] **Step 1: Provide a real test clip path via env**

Place a short (~30 s) real audio file somewhere and note its absolute path. It will be passed as `PROBE_AUDIO`.

- [ ] **Step 2: Write the probe that transcribes + aligns**

`Sources/asr-probe/main.swift`:
```swift
import Foundation
import AVFoundation
import Qwen3ASR

// Reconciled speech-swift API (update this comment if the compiler disagrees):
//   Qwen3ASRModel.fromPretrained() async throws -> Qwen3ASRModel
//   model.transcribe(audio: [Float], sampleRate: Int) -> String
//   Qwen3ForcedAligner.fromPretrained() async throws -> Qwen3ForcedAligner
//   aligner.align(audio: [Float], text: String, sampleRate: Int) -> [AlignedWord]
//   AlignedWord: .text, .startTime, .endTime

func loadMono16k(_ url: URL) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    let src = file.processingFormat
    let frames = AVAudioFrameCount(file.length)
    let srcBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: frames)!
    try file.read(into: srcBuf)
    let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    let conv = AVAudioConverter(from: src, to: dst)!
    let cap = AVAudioFrameCount(Double(frames) * 16000.0 / src.sampleRate) + 1024
    let dstBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: cap)!
    var fed = false
    _ = conv.convert(to: dstBuf, error: nil) { _, status in
        if fed { status.pointee = .noDataNow; return nil }
        fed = true; status.pointee = .haveData; return srcBuf
    }
    let ch = dstBuf.floatChannelData![0]
    return Array(UnsafeBufferPointer(start: ch, count: Int(dstBuf.frameLength)))
}

let path = ProcessInfo.processInfo.environment["PROBE_AUDIO"] ?? ""
let url = URL(fileURLWithPath: path)
let samples = try loadMono16k(url)
print("loaded \(samples.count) samples (\(Double(samples.count)/16000.0) s)")

let model = try await Qwen3ASRModel.fromPretrained()
let text = model.transcribe(audio: samples, sampleRate: 16000)
print("TEXT:\n\(text)")

let aligner = try await Qwen3ForcedAligner.fromPretrained()
let aligned = aligner.align(audio: samples, text: text, sampleRate: 16000)
for w in aligned.prefix(10) {
    print("[\(w.startTime)–\(w.endTime)] \(w.text)")
}
```

- [ ] **Step 3: Build; fix signatures against the compiler**

Run: `swift build`
Expected: compiles. If it does not, adjust the calls to match the real API and update the reconciliation comment. Do not proceed until it builds.

- [ ] **Step 4: Run against the real clip (downloads models on first run)**

Run: `PROBE_AUDIO=/absolute/path/to/clip.wav swift run asr-probe`
Expected: prints the sample count, a non-empty `TEXT:` block, and up to 10 `[start–end] word` lines. First run downloads weights to `~/Library/Caches/qwen3-speech/`.

- [ ] **Step 5: Commit**

```bash
git add Sources/asr-probe/main.swift
git commit -m "feat(probe): transcribe + align a real clip via speech-swift"
```

---

### Task 3: Domain types + TranscriptionProvider protocol

**Files:**
- Delete: `Sources/ReplikaCore/Placeholder.swift`
- Create: `Sources/ReplikaCore/TranscriptTypes.swift`
- Create: `Sources/ReplikaCore/TranscriptionProvider.swift`
- Create: `Tests/ReplikaCoreTests/TranscriptTypesTests.swift`
- Modify: `Package.swift` (add `ReplikaCoreTests` test target)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Word(text: String, start: Double, end: Double)`
  - `Segment(text: String, start: Double, end: Double, words: [Word], speaker: Int? = nil)` — `speaker` is `var`
  - `SpeakerTag(speaker: Int, start: Double, end: Double)`
  - `Transcript(segments: [Segment], language: String? = nil)`
  - `enum TranscriptEvent { case progress(Double); case committed(Segment); case speaker(SpeakerTag); case done(Transcript) }`
  - `enum AudioSource { case file(URL) }`
  - `ProviderCaps(diarization: Bool, wordTimestamps: Bool, streaming: Bool)`
  - `enum Quant: String { case q4; case q8 }`
  - `TranscribeOptions(language: String = "auto", quant: Quant = .q4, diarize: Bool = true, contextHint: String? = nil)`
  - `protocol TranscriptionProvider: Sendable { var capabilities: ProviderCaps { get }; func transcribe(_:options:) -> AsyncThrowingStream<TranscriptEvent, Error>; func cancel() }`

- [ ] **Step 1: Add the test target to `Package.swift`**

Add to the `targets:` array:
```swift
        .testTarget(name: "ReplikaCoreTests", dependencies: ["ReplikaCore"]),
```

- [ ] **Step 2: Write the failing test**

`Tests/ReplikaCoreTests/TranscriptTypesTests.swift`:
```swift
import Testing
import Foundation
@testable import ReplikaCore

@Test func segmentDefaultsToNoSpeaker() {
    let seg = Segment(text: "hi", start: 0, end: 1, words: [Word(text: "hi", start: 0, end: 1)])
    #expect(seg.speaker == nil)
    #expect(seg.words.count == 1)
}

@Test func transcribeOptionsDefaults() {
    let o = TranscribeOptions()
    #expect(o.language == "auto")
    #expect(o.quant == .q4)
    #expect(o.diarize == true)
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `swift test --filter ReplikaCoreTests`
Expected: FAIL — `Segment`/`Word`/`TranscribeOptions` not found.

- [ ] **Step 4: Delete the placeholder and write the types**

Delete `Sources/ReplikaCore/Placeholder.swift`.

`Sources/ReplikaCore/TranscriptTypes.swift`:
```swift
import Foundation

public struct Word: Sendable, Equatable {
    public let text: String
    public let start: Double
    public let end: Double
    public init(text: String, start: Double, end: Double) {
        self.text = text; self.start = start; self.end = end
    }
}

public struct Segment: Sendable, Equatable {
    public let text: String
    public let start: Double
    public let end: Double
    public let words: [Word]
    public var speaker: Int?
    public init(text: String, start: Double, end: Double, words: [Word], speaker: Int? = nil) {
        self.text = text; self.start = start; self.end = end
        self.words = words; self.speaker = speaker
    }
}

public struct SpeakerTag: Sendable, Equatable {
    public let speaker: Int
    public let start: Double
    public let end: Double
    public init(speaker: Int, start: Double, end: Double) {
        self.speaker = speaker; self.start = start; self.end = end
    }
}

public struct Transcript: Sendable, Equatable {
    public let segments: [Segment]
    public let language: String?
    public init(segments: [Segment], language: String? = nil) {
        self.segments = segments; self.language = language
    }
}

public enum TranscriptEvent: Sendable {
    case progress(Double)
    case committed(Segment)
    case speaker(SpeakerTag)
    case done(Transcript)
}
```

`Sources/ReplikaCore/TranscriptionProvider.swift`:
```swift
import Foundation

public enum AudioSource: Sendable {
    case file(URL)
}

public struct ProviderCaps: Sendable, Equatable {
    public let diarization: Bool
    public let wordTimestamps: Bool
    public let streaming: Bool
    public init(diarization: Bool, wordTimestamps: Bool, streaming: Bool) {
        self.diarization = diarization
        self.wordTimestamps = wordTimestamps
        self.streaming = streaming
    }
}

public enum Quant: String, Sendable {
    case q4
    case q8
}

public struct TranscribeOptions: Sendable {
    public let language: String
    public let quant: Quant
    public let diarize: Bool
    public let contextHint: String?
    public init(language: String = "auto", quant: Quant = .q4,
                diarize: Bool = true, contextHint: String? = nil) {
        self.language = language; self.quant = quant
        self.diarize = diarize; self.contextHint = contextHint
    }
}

public protocol TranscriptionProvider: Sendable {
    var capabilities: ProviderCaps { get }
    func transcribe(_ audio: AudioSource,
                    options: TranscribeOptions) -> AsyncThrowingStream<TranscriptEvent, Error>
    func cancel()
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `swift test --filter ReplikaCoreTests`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Package.swift Sources/ReplikaCore Tests/ReplikaCoreTests
git commit -m "feat(core): domain types and TranscriptionProvider protocol"
```

---

### Task 4: AudioLoader — decode any file to 16 kHz mono

**Files:**
- Create: `Sources/ReplikaCore/AudioLoader.swift`
- Create: `Tests/ReplikaCoreTests/AudioLoaderTests.swift`

**Interfaces:**
- Consumes: nothing.
- Produces: `enum AudioLoader { static func loadMono16k(_ url: URL) throws -> [Float] }`; `enum AudioLoaderError: Error { case cannotOpen(URL); case conversionFailed }`.

- [ ] **Step 1: Write the failing test**

`Tests/ReplikaCoreTests/AudioLoaderTests.swift`:
```swift
import Testing
import Foundation
import AVFoundation
@testable import ReplikaCore

private func writeSineWav(_ url: URL, seconds: Double, sampleRate: Double) throws {
    let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
                            channels: 1, interleaved: false)!
    let file = try AVAudioFile(forWriting: url, settings: fmt.settings)
    let frames = AVAudioFrameCount(seconds * sampleRate)
    let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: frames)!
    buf.frameLength = frames
    let ptr = buf.floatChannelData![0]
    for i in 0..<Int(frames) {
        ptr[i] = Float(sin(2.0 * .pi * 440.0 * Double(i) / sampleRate)) * 0.5
    }
    try file.write(from: buf)
}

@Test func loadsAndResamplesToMono16k() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("sine_\(UUID().uuidString).wav")
    try writeSineWav(url, seconds: 1.0, sampleRate: 44100)
    defer { try? FileManager.default.removeItem(at: url) }

    let samples = try AudioLoader.loadMono16k(url)
    // ~16000 samples for 1 s, allow small converter slack
    #expect(abs(samples.count - 16000) < 400)
    #expect(samples.contains { $0 != 0 })
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter AudioLoaderTests`
Expected: FAIL — `AudioLoader` not found.

- [ ] **Step 3: Implement `AudioLoader`**

`Sources/ReplikaCore/AudioLoader.swift`:
```swift
import AVFoundation

public enum AudioLoaderError: Error {
    case cannotOpen(URL)
    case conversionFailed
}

public enum AudioLoader {
    /// Decode any AVFoundation-supported file to 16 kHz mono Float32 samples.
    public static func loadMono16k(_ url: URL) throws -> [Float] {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) }
        catch { throw AudioLoaderError.cannotOpen(url) }

        let src = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard let srcBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: frames) else {
            throw AudioLoaderError.conversionFailed
        }
        try file.read(into: srcBuf)

        guard let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                      channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: src, to: dst) else {
            throw AudioLoaderError.conversionFailed
        }

        let cap = AVAudioFrameCount(Double(frames) * 16000.0 / src.sampleRate) + 1024
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: cap) else {
            throw AudioLoaderError.conversionFailed
        }

        var fed = false
        var convError: NSError?
        let status = conv.convert(to: dstBuf, error: &convError) { _, outStatus in
            if fed { outStatus.pointee = .noDataNow; return nil }
            fed = true
            outStatus.pointee = .haveData
            return srcBuf
        }
        if status == .error || convError != nil {
            throw AudioLoaderError.conversionFailed
        }

        guard let ch = dstBuf.floatChannelData else { throw AudioLoaderError.conversionFailed }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(dstBuf.frameLength)))
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter AudioLoaderTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/ReplikaCore/AudioLoader.swift Tests/ReplikaCoreTests/AudioLoaderTests.swift
git commit -m "feat(core): AudioLoader decodes files to 16 kHz mono"
```

---

### Task 5: SegmentBuilder — group words into segments by pause

**Files:**
- Create: `Sources/ReplikaCore/SegmentBuilder.swift`
- Create: `Tests/ReplikaCoreTests/SegmentBuilderTests.swift`

**Interfaces:**
- Consumes: `Word`, `Segment` (Task 3).
- Produces: `enum SegmentBuilder { static func build(words: [Word], pauseThreshold: Double = 0.6) -> [Segment] }`.

- [ ] **Step 1: Write the failing test**

`Tests/ReplikaCoreTests/SegmentBuilderTests.swift`:
```swift
import Testing
@testable import ReplikaCore

@Test func splitsOnLongPause() {
    let words = [
        Word(text: "a", start: 0.0, end: 0.3),
        Word(text: "b", start: 0.35, end: 0.6),   // small gap -> same segment
        Word(text: "c", start: 2.0, end: 2.3)      // >0.6 s gap -> new segment
    ]
    let segs = SegmentBuilder.build(words: words, pauseThreshold: 0.6)
    #expect(segs.count == 2)
    #expect(segs[0].text == "a b")
    #expect(segs[1].text == "c")
    #expect(segs[0].start == 0.0)
    #expect(segs[0].end == 0.6)
}

@Test func emptyWordsYieldNoSegments() {
    #expect(SegmentBuilder.build(words: []).isEmpty)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter SegmentBuilderTests`
Expected: FAIL — `SegmentBuilder` not found.

- [ ] **Step 3: Implement `SegmentBuilder`**

`Sources/ReplikaCore/SegmentBuilder.swift`:
```swift
public enum SegmentBuilder {
    /// Group consecutive words into segments, breaking when the silent gap
    /// between two words exceeds `pauseThreshold` seconds.
    public static func build(words: [Word], pauseThreshold: Double = 0.6) -> [Segment] {
        guard let first = words.first else { return [] }
        var segments: [Segment] = []
        var current: [Word] = [first]
        for w in words.dropFirst() {
            let gap = w.start - (current.last?.end ?? w.start)
            if gap > pauseThreshold {
                segments.append(makeSegment(current))
                current = [w]
            } else {
                current.append(w)
            }
        }
        segments.append(makeSegment(current))
        return segments
    }

    private static func makeSegment(_ words: [Word]) -> Segment {
        Segment(text: words.map(\.text).joined(separator: " "),
                start: words.first?.start ?? 0,
                end: words.last?.end ?? 0,
                words: words,
                speaker: nil)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter SegmentBuilderTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/ReplikaCore/SegmentBuilder.swift Tests/ReplikaCoreTests/SegmentBuilderTests.swift
git commit -m "feat(core): SegmentBuilder groups words into segments by pause"
```

---

### Task 6: SpeakerMerger — assign speakers to segments by overlap

**Files:**
- Create: `Sources/ReplikaCore/SpeakerMerger.swift`
- Create: `Tests/ReplikaCoreTests/SpeakerMergerTests.swift`

**Interfaces:**
- Consumes: `Segment`, `SpeakerTag` (Task 3).
- Produces: `enum SpeakerMerger { static func merge(segments: [Segment], tags: [SpeakerTag]) -> [Segment] }`.

- [ ] **Step 1: Write the failing test**

`Tests/ReplikaCoreTests/SpeakerMergerTests.swift`:
```swift
import Testing
@testable import ReplikaCore

@Test func assignsSpeakerWithMaxOverlap() {
    let seg = Segment(text: "hello", start: 1.0, end: 3.0,
                      words: [Word(text: "hello", start: 1.0, end: 3.0)])
    let tags = [
        SpeakerTag(speaker: 0, start: 0.0, end: 1.4),  // 0.4 s overlap
        SpeakerTag(speaker: 1, start: 1.4, end: 3.0)   // 1.6 s overlap -> wins
    ]
    let merged = SpeakerMerger.merge(segments: [seg], tags: tags)
    #expect(merged[0].speaker == 1)
}

@Test func noTagsLeavesSpeakerNil() {
    let seg = Segment(text: "x", start: 0, end: 1, words: [Word(text: "x", start: 0, end: 1)])
    #expect(SpeakerMerger.merge(segments: [seg], tags: []).first?.speaker == nil)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter SpeakerMergerTests`
Expected: FAIL — `SpeakerMerger` not found.

- [ ] **Step 3: Implement `SpeakerMerger`**

`Sources/ReplikaCore/SpeakerMerger.swift`:
```swift
public enum SpeakerMerger {
    /// Assign each segment the speaker whose diarization tags overlap it most.
    public static func merge(segments: [Segment], tags: [SpeakerTag]) -> [Segment] {
        guard !tags.isEmpty else { return segments }
        return segments.map { seg in
            var best: Int?
            var bestOverlap = 0.0
            for tag in tags {
                let ov = overlap(seg.start, seg.end, tag.start, tag.end)
                if ov > bestOverlap {
                    bestOverlap = ov
                    best = tag.speaker
                }
            }
            var copy = seg
            copy.speaker = best
            return copy
        }
    }

    private static func overlap(_ a0: Double, _ a1: Double,
                                _ b0: Double, _ b1: Double) -> Double {
        max(0, min(a1, b1) - max(a0, b0))
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter SpeakerMergerTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/ReplikaCore/SpeakerMerger.swift Tests/ReplikaCoreTests/SpeakerMergerTests.swift
git commit -m "feat(core): SpeakerMerger labels segments by diarization overlap"
```

---

### Task 7: ProviderRegistry — factory/registry

**Files:**
- Create: `Sources/ReplikaCore/ProviderRegistry.swift`
- Create: `Tests/ReplikaCoreTests/ProviderRegistryTests.swift`

**Interfaces:**
- Consumes: `TranscriptionProvider`, `ProviderCaps`, `TranscriptEvent`, `AudioSource`, `TranscribeOptions` (Task 3).
- Produces: `struct ProviderRegistry { typealias Factory = @Sendable () -> TranscriptionProvider; init(); mutating func register(_ name: String, _ factory: @escaping Factory); func make(_ name: String) -> TranscriptionProvider?; var names: [String] }`.

- [ ] **Step 1: Write the failing test**

`Tests/ReplikaCoreTests/ProviderRegistryTests.swift`:
```swift
import Testing
import Foundation
@testable import ReplikaCore

private struct StubProvider: TranscriptionProvider {
    let capabilities = ProviderCaps(diarization: false, wordTimestamps: false, streaming: false)
    func transcribe(_ audio: AudioSource,
                    options: TranscribeOptions) -> AsyncThrowingStream<TranscriptEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func cancel() {}
}

@Test func registersAndMakesProvider() {
    var reg = ProviderRegistry()
    reg.register("stub") { StubProvider() }
    #expect(reg.make("stub") != nil)
    #expect(reg.make("missing") == nil)
    #expect(reg.names == ["stub"])
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter ProviderRegistryTests`
Expected: FAIL — `ProviderRegistry` not found.

- [ ] **Step 3: Implement `ProviderRegistry`**

`Sources/ReplikaCore/ProviderRegistry.swift`:
```swift
public struct ProviderRegistry {
    public typealias Factory = @Sendable () -> TranscriptionProvider

    private var factories: [String: Factory] = [:]

    public init() {}

    public mutating func register(_ name: String, _ factory: @escaping Factory) {
        factories[name] = factory
    }

    public func make(_ name: String) -> TranscriptionProvider? {
        factories[name]?()
    }

    public var names: [String] { factories.keys.sorted() }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `swift test --filter ProviderRegistryTests`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/ReplikaCore/ProviderRegistry.swift Tests/ReplikaCoreTests/ProviderRegistryTests.swift
git commit -m "feat(core): ProviderRegistry factory/registry"
```

---

### Task 8: SpeechSwiftProvider — wrap speech-swift behind the protocol

**Files:**
- Delete: `Sources/SpeechSwiftProvider/Placeholder.swift`
- Create: `Sources/SpeechSwiftProvider/SpeechSwiftProvider.swift`
- Create: `Tests/SpeechSwiftProviderTests/SmokeTests.swift`
- Create: `Tests/SpeechSwiftProviderTests/Resources/sample_short.wav` (a real ~20–30 s multi-speaker clip)
- Modify: `Package.swift` (add `SpeechSwiftProviderTests` target with the resource)

**Interfaces:**
- Consumes: `AudioLoader`, `SegmentBuilder`, `SpeakerMerger`, all Task-3 types; `speech-swift` API confirmed in Task 2.
- Produces: `final class SpeechSwiftProvider: TranscriptionProvider` (public `init()`); `enum ProviderError: Error { case unsupportedSource }`.

- [ ] **Step 1: Add the test target to `Package.swift`**

Add to the `targets:` array:
```swift
        .testTarget(
            name: "SpeechSwiftProviderTests",
            dependencies: ["SpeechSwiftProvider", "ReplikaCore"],
            resources: [.copy("Resources/sample_short.wav")]
        ),
```

- [ ] **Step 2: Add the resource clip**

Copy a real ~20–30 s clip with at least two speakers to
`Tests/SpeechSwiftProviderTests/Resources/sample_short.wav`.

- [ ] **Step 3: Write the smoke test (gated by env so `swift test` stays green offline)**

`Tests/SpeechSwiftProviderTests/SmokeTests.swift`:
```swift
import Testing
import Foundation
import ReplikaCore
@testable import SpeechSwiftProvider

@Test(.enabled(if: ProcessInfo.processInfo.environment["RUN_MODEL_TESTS"] == "1"))
func smokeProducesLabeledSegments() async throws {
    let url = Bundle.module.url(forResource: "sample_short", withExtension: "wav")!
    let provider = SpeechSwiftProvider()

    var done: Transcript?
    var sawSpeaker = false
    for try await ev in provider.transcribe(.file(url), options: TranscribeOptions(diarize: true)) {
        switch ev {
        case .speaker: sawSpeaker = true
        case .done(let t): done = t
        default: break
        }
    }

    let transcript = try #require(done)
    #expect(!transcript.segments.isEmpty)
    #expect(transcript.segments.allSatisfy { !$0.text.isEmpty })
    #expect(sawSpeaker)
    for seg in transcript.segments where seg.words.count > 1 {
        for i in 1..<seg.words.count {
            #expect(seg.words[i].start >= seg.words[i - 1].start)  // monotonic
            #expect(seg.words[i].end <= seg.end + 0.001)           // within segment
        }
    }
}
```

- [ ] **Step 4: Run the test to verify it fails (compile failure — provider not written)**

Run: `RUN_MODEL_TESTS=1 swift test --filter SpeechSwiftProviderTests`
Expected: FAIL — `SpeechSwiftProvider` not found.

- [ ] **Step 5: Delete the placeholder and implement the provider**

Delete `Sources/SpeechSwiftProvider/Placeholder.swift`.

`Sources/SpeechSwiftProvider/SpeechSwiftProvider.swift`:
```swift
import Foundation
import os
import ReplikaCore
import Qwen3ASR
import SpeechVAD

public enum ProviderError: Error {
    case unsupportedSource
}

public final class SpeechSwiftProvider: TranscriptionProvider, @unchecked Sendable {
    public let capabilities = ProviderCaps(diarization: true, wordTimestamps: true, streaming: false)

    private let logger = Logger(subsystem: "com.replika.spike", category: "SpeechSwiftProvider")
    private let lock = NSLock()
    private var task: Task<Void, Never>?

    public init() {}

    public func transcribe(_ audio: AudioSource,
                           options: TranscribeOptions) -> AsyncThrowingStream<TranscriptEvent, Error> {
        AsyncThrowingStream { continuation in
            let work = Task {
                do {
                    guard case let .file(url) = audio else { throw ProviderError.unsupportedSource }
                    let samples = try AudioLoader.loadMono16k(url)
                    continuation.yield(.progress(0.1))

                    // Diarization (optional). NOTE: quant selection (options.quant)
                    // maps to a fromPretrained() variant argument — confirm against Task 2.
                    var tags: [SpeakerTag] = []
                    if options.diarize {
                        let diarizer = try await DiarizationPipeline.fromPretrained()
                        try Task.checkCancellation()
                        let dsegs = diarizer.diarize(audio: samples, sampleRate: 16000)
                        tags = dsegs.map { SpeakerTag(speaker: $0.speakerId,
                                                      start: $0.startTime, end: $0.endTime) }
                        for tag in tags { continuation.yield(.speaker(tag)) }
                    }
                    continuation.yield(.progress(0.4))

                    // ASR
                    let model = try await Qwen3ASRModel.fromPretrained()
                    try Task.checkCancellation()
                    let text = model.transcribe(audio: samples, sampleRate: 16000)
                    continuation.yield(.progress(0.7))

                    // Word timestamps
                    let aligner = try await Qwen3ForcedAligner.fromPretrained()
                    try Task.checkCancellation()
                    let aligned = aligner.align(audio: samples, text: text, sampleRate: 16000)
                    let words = aligned.map { Word(text: $0.text, start: $0.startTime, end: $0.endTime) }

                    // Segments + speakers
                    let base = SegmentBuilder.build(words: words)
                    let merged = SpeakerMerger.merge(segments: base, tags: tags)
                    for seg in merged { continuation.yield(.committed(seg)) }

                    continuation.yield(.done(Transcript(segments: merged, language: options.language)))
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: CancellationError())
                } catch {
                    self.logger.error("transcribe failed: \(error.localizedDescription, privacy: .public)")
                    continuation.finish(throwing: error)
                }
            }
            lock.lock(); self.task = work; lock.unlock()
            continuation.onTermination = { _ in work.cancel() }
        }
    }

    public func cancel() {
        lock.lock(); let t = task; lock.unlock()
        t?.cancel()
    }
}
```

- [ ] **Step 6: Build without models to confirm it compiles**

Run: `swift build`
Expected: compiles. Reconcile any `speech-swift` signature mismatch against Task 2's probe.

- [ ] **Step 7: Run the gated smoke test (downloads models)**

Run: `RUN_MODEL_TESTS=1 swift test --filter SpeechSwiftProviderTests`
Expected: PASS — non-empty labeled segments, at least one speaker event, monotonic word timestamps.

- [ ] **Step 8: Confirm the offline suite is still green**

Run: `swift test`
Expected: PASS — smoke test is skipped (env var unset), all `ReplikaCore` tests pass.

- [ ] **Step 9: Commit**

```bash
git add Package.swift Sources/SpeechSwiftProvider Tests/SpeechSwiftProviderTests
git commit -m "feat(provider): SpeechSwiftProvider wraps Qwen3-ASR + diarization behind protocol"
```

---

### Task 9: CLI + benchmark harness

**Files:**
- Create: `Sources/replika-spike/main.swift`
- Create: `Sources/replika-spike/Bench.swift`
- Modify: `Package.swift` (add `replika-spike` executable target)

**Interfaces:**
- Consumes: `SpeechSwiftProvider`, `ProviderRegistry`, `AudioLoader`, all Task-3 types.
- Produces: an executable `replika-spike` accepting `<audio-file>` and flags `--quant q4|q8`, `--no-diarize`; prints labeled transcript + a Markdown benchmark row.

- [ ] **Step 1: Add the executable target to `Package.swift`**

Add to the `targets:` array:
```swift
        .executableTarget(
            name: "replika-spike",
            dependencies: ["ReplikaCore", "SpeechSwiftProvider"]
        ),
```

- [ ] **Step 2: Write `Bench.swift` (RTF + peak memory)**

`Sources/replika-spike/Bench.swift`:
```swift
import Foundation
import Darwin

/// Current physical memory footprint of this process, in bytes.
func currentFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4
    let kr = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

struct BenchResult {
    let clip: String
    let quant: String
    let rtf: Double
    let peakMB: Double
    let loadAndRunSec: Double

    var markdownRow: String {
        String(format: "| %@ | %@ | %.2f | %.0f | %.1f |",
               clip, quant, rtf, peakMB, loadAndRunSec)
    }

    static var markdownHeader: String {
        "| clip | quant | RTF | peak RAM (MB) | wall (s) |\n|---|---|---|---|---|"
    }
}
```

- [ ] **Step 3: Write `main.swift` (drive the provider + print)**

`Sources/replika-spike/main.swift`:
```swift
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
        if let v = args.first, let q = Quant(rawValue: v) { quant = q; args.removeFirst() }
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
let samples = try AudioLoader.loadMono16k(url)
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

for try await ev in provider.transcribe(.file(url), options: options) {
    switch ev {
    case .committed(let seg):
        let who = seg.speaker.map { "S\($0)" } ?? "S?"
        print("[\(fmt(seg.start))–\(fmt(seg.end))] \(who): \(seg.text)")
    case .done:
        break
    default:
        break
    }
}

let wall = Double((clock.now - start).components.seconds) +
           Double((clock.now - start).components.attoseconds) / 1e18
poller.cancel()

let result = BenchResult(
    clip: clipName,
    quant: quant.rawValue,
    rtf: audioSeconds / max(wall, 0.0001),
    peakMB: Double(peak.value) / 1_048_576.0,
    loadAndRunSec: wall
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
```

- [ ] **Step 4: Build**

Run: `swift build`
Expected: compiles.

- [ ] **Step 5: Run against a real clip**

Run: `swift run replika-spike /absolute/path/to/clip.wav --quant q4`
Expected: prints labeled segments (`[start–end] S0: …`) followed by a one-row Markdown table with RTF, peak RAM, wall time.

- [ ] **Step 6: Commit**

```bash
git add Package.swift Sources/replika-spike
git commit -m "feat(cli): replika-spike driver with RTF + peak-memory bench"
```

---

### Task 10: Run the benchmark on real clips + write findings

Turns the spike into a decision: fills the benchmark table and records the verdict.

**Files:**
- Create: `docs/superpowers/specs/2026-07-15-mlx-asr-feasibility-spike-FINDINGS.md`

**Interfaces:**
- Consumes: `replika-spike` CLI (Task 9).
- Produces: a committed findings document with the filled table and a go/no-go verdict on the "wrap speech-swift" strategy.

- [ ] **Step 1: Run the matrix (quant × clip) on your own RU / multi-speaker clips**

Run, for each clip and each quant:
```bash
swift run replika-spike /path/to/ru_clip.wav          --quant q4
swift run replika-spike /path/to/ru_clip.wav          --quant q8
swift run replika-spike /path/to/multispeaker.wav     --quant q4
swift run replika-spike /path/to/multispeaker.wav     --quant q8
```
Record each Markdown row and eyeball the transcript quality (RU correctness, speaker labels look right, word timestamps line up on playback).

- [ ] **Step 2: Write the findings document**

`docs/superpowers/specs/2026-07-15-mlx-asr-feasibility-spike-FINDINGS.md`:
```markdown
# MLX-ASR Feasibility Spike — Findings (2026-07-15)

## Benchmark

| clip | quant | RTF | peak RAM (MB) | wall (s) |
|---|---|---|---|---|
| <paste rows from replika-spike> |

## Quality (by ear)
- RU correctness: <notes>
- Speaker labels: <notes>
- Word-timestamp alignment on playback: <notes>

## Build / runtime reality
- Builds as SwiftPM CLI on Swift 6 / Metal Toolchain: <yes/no + notes>
- Sortformer diarization runs from CLI (no app bundle): <yes/no + notes>
- speech-swift API differences vs the plan's assumptions: <list>

## Verdict
- Wrap-speech-swift strategy confirmed for the product? <yes/no>
- Gaps to carry into the next slice: <model-cache relocation, live mode, …>
- Recommended default: <1.7B q4 | q8> based on RTF/RAM/quality above.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-15-mlx-asr-feasibility-spike-FINDINGS.md
git commit -m "docs: MLX-ASR feasibility spike findings and verdict"
```

---

## Self-Review Notes

- **Spec coverage:** acceptance criteria 1–2 → Tasks 2, 8; criterion 3 (diarization+merge) → Tasks 6, 8; criterion 4 (protocol boundary) → Tasks 3, 8, 9; criterion 5 (measured 4-bit/8-bit) → Tasks 9, 10; criterion 6 (quality sanity) → Task 10. File structure from spec §4 mapped 1:1 onto targets.
- **Known external-API risk:** every `speech-swift` call site is reconciled at Task 2 and re-checked at Task 8 Step 6; quant→`fromPretrained` mapping is explicitly flagged as reconcile-at-build (not silently assumed).
- **Out of scope stays out:** no live mode, no model-manager/cache relocation, no remote provider — all deferred to later slices per spec §10.
