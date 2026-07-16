# Long-form VAD Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-shot `Qwen3ASRModel.transcribe(whole-file)` call with VAD-guided chunking so multi-minute calls produce correct, speaker-labeled, timestamped segments.

**Architecture:** Pure, model-free planning/stitching logic lives in `ReplikaCore` (`ChunkPlanner`, `WordStitcher`, `ChunkConfig`); the model-bound orchestration lives in `SpeechSwiftProvider`. The provider runs Silero VAD over the full buffer, plans bounded windows, transcribes+aligns each window (each ≤ `maxChunkSeconds`, safely under the 448-token ASR cap and the ~270 s aligner plateau), offsets word timestamps to absolute time, stitches them, and feeds the existing `SegmentBuilder → SpeakerMerger` downstream unchanged.

**Tech Stack:** Swift 6 (SwiftPM), `speech-swift` (Qwen3ASR + SpeechVAD modules), Swift Testing, MLX/CoreML on Apple Silicon.

## Global Constraints

- Swift 6, `swift-tools-version: 6.0`. Library targets (`ReplikaCore`, `SpeechSwiftProvider`) MUST compile warning-free.
- Test framework is **Swift Testing** (`import Testing`, `@Test`, `#expect`, `#require`, `.enabled(if:)`). Not XCTest.
- PRIVACY (BINDING): never commit audio files (`*.m4a *.wav *.mp3 *.flac *.aac` are gitignored). Model-running tests read the clip path from `SMOKE_AUDIO` and only run when `RUN_MODEL_TESTS == "1"` AND `SMOKE_AUDIO` is a non-empty path. Zero transcript text in logs or assertions — counts/metrics only.
- Logging: `print()` is allowed ONLY in the `replika-spike` CLI target (it is the UI). Library targets use `os.Logger`.
- `speech-swift` is pinned via `Package.resolved` — do not bump it. Verified upstream signatures used by this plan:
  - `SileroVADModel.fromPretrained(modelId:engine:cacheDir:offlineMode:progressHandler:)`, `SileroVADModel.detectSpeech(audio:sampleRate:config:) -> [SpeechSegment]`, `SileroVADModel.defaultModelId`.
  - `Qwen3ASRModel.fromPretrained(modelId:cacheDir:offlineMode:progressHandler:)`, `.transcribe(audio:sampleRate:language:maxTokens:context:) -> String`.
  - `Qwen3ForcedAligner.fromPretrained(modelId:cacheDir:offlineMode:progressHandler:)`, `.align(audio:text:sampleRate:language:) -> [AlignedWord]` (`AlignedWord{text:String,startTime:Float,endTime:Float}`).
  - `SortformerDiarizer.fromPretrained(modelId:cacheDir:offlineMode:config:computeUnits:progressHandler:)`, `.diarize(audio:sampleRate:config:) -> DiarizationResult`, `.defaultModelId`.
  - `HuggingFaceDownloader.getCacheDirectory(for:basePath:cacheDirName:) throws -> URL` (AudioCommon).
  - `VADConfig(onset:offset:minSpeechDuration:minSilenceDuration:windowDuration:stepRatio:)`, `VADConfig.sileroDefault` (all `Float`).
- Running model tests (E2E) requires the MLX metallib present in `.build` (and copied into the xctest bundle) — see FINDINGS §3. Build first; if a test aborts with `MLX error: Failed to load the default metallib`, that is the metallib gap, not a code bug.

Spec: `docs/superpowers/specs/2026-07-16-longform-vad-chunking-design.md`

---

## File Structure

- Create `Sources/ReplikaCore/ChunkConfig.swift` — `VadTier` enum, `ChunkConfig` struct.
- Create `Sources/ReplikaCore/ChunkPlanner.swift` — `ChunkWindow` struct, `ChunkPlanner.plan`.
- Create `Sources/ReplikaCore/WordStitcher.swift` — `WordStitcher.stitch`.
- Modify `Sources/ReplikaCore/TranscriptionProvider.swift` — add `chunk: ChunkConfig` to `TranscribeOptions`.
- Modify `Sources/ReplikaCore/AudioLoader.swift` — error-surface hardening.
- Modify `Sources/SpeechSwiftProvider/SpeechSwiftProvider.swift` — VAD-chunk loop + language plumbing + cache relocation.
- Modify `Sources/replika-spike/Bench.swift` — add `segments` to `BenchResult`.
- Modify `Sources/replika-spike/main.swift` — count committed segments, pass to `BenchResult`.
- Create `Tests/ReplikaCoreTests/ChunkPlannerTests.swift`, `Tests/ReplikaCoreTests/WordStitcherTests.swift`.
- Modify `Tests/ReplikaCoreTests/AudioLoaderTests.swift` — add error-path tests.
- Modify `Tests/SpeechSwiftProviderTests/SmokeTests.swift` — add long-form E2E regression test.

---

### Task 1: ChunkConfig + VadTier + wire into TranscribeOptions

**Files:**
- Create: `Sources/ReplikaCore/ChunkConfig.swift`
- Modify: `Sources/ReplikaCore/TranscriptionProvider.swift:23-33`
- Test: `Tests/ReplikaCoreTests/ChunkConfigTests.swift` (create)

**Interfaces:**
- Produces: `enum VadTier: String, Sendable, Equatable { case silero }`;
  `struct ChunkConfig: Sendable, Equatable` with `init(maxChunkSeconds: Double = 10.0, overlapSeconds: Double = 0.5, vadTier: VadTier = .silero, minSpeechSeconds: Double = 0.25)` and stored `let` properties of the same names;
  `TranscribeOptions.chunk: ChunkConfig` (new stored property, default `ChunkConfig()`).

- [ ] **Step 1: Write the failing test**

Create `Tests/ReplikaCoreTests/ChunkConfigTests.swift`:

```swift
import Testing
@testable import ReplikaCore

@Test func chunkConfigDefaults() {
    let c = ChunkConfig()
    #expect(c.maxChunkSeconds == 10.0)
    #expect(c.overlapSeconds == 0.5)
    #expect(c.vadTier == .silero)
    #expect(c.minSpeechSeconds == 0.25)
}

@Test func transcribeOptionsCarriesChunkConfig() {
    let opts = TranscribeOptions(chunk: ChunkConfig(maxChunkSeconds: 5.0))
    #expect(opts.chunk.maxChunkSeconds == 5.0)
    // Default still applies when omitted.
    #expect(TranscribeOptions().chunk.maxChunkSeconds == 10.0)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `swift test --filter chunkConfigDefaults`
Expected: FAILS TO COMPILE — `cannot find 'ChunkConfig' in scope`.

- [ ] **Step 3: Create ChunkConfig.swift**

Create `Sources/ReplikaCore/ChunkConfig.swift`:

```swift
import Foundation

/// Which VAD backend segments the audio before per-chunk transcription.
public enum VadTier: String, Sendable, Equatable {
    case silero
}

/// Tuning for long-form chunked transcription. `maxChunkSeconds` bounds each
/// ASR/aligner call well under the 448-token cap and ~270 s aligner plateau;
/// `overlapSeconds` applies only when a single continuous speech span is
/// force-split (no silence to cut at).
public struct ChunkConfig: Sendable, Equatable {
    public let maxChunkSeconds: Double
    public let overlapSeconds: Double
    public let vadTier: VadTier
    public let minSpeechSeconds: Double

    public init(maxChunkSeconds: Double = 10.0,
                overlapSeconds: Double = 0.5,
                vadTier: VadTier = .silero,
                minSpeechSeconds: Double = 0.25) {
        self.maxChunkSeconds = maxChunkSeconds
        self.overlapSeconds = overlapSeconds
        self.vadTier = vadTier
        self.minSpeechSeconds = minSpeechSeconds
    }
}
```

- [ ] **Step 4: Wire `chunk` into TranscribeOptions**

In `Sources/ReplikaCore/TranscriptionProvider.swift`, replace the `TranscribeOptions` struct (currently lines 23-33) with:

```swift
public struct TranscribeOptions: Sendable {
    public let language: String
    public let quant: Quant
    public let diarize: Bool
    public let contextHint: String?
    public let chunk: ChunkConfig
    public init(language: String = "auto", quant: Quant = .q4,
                diarize: Bool = true, contextHint: String? = nil,
                chunk: ChunkConfig = ChunkConfig()) {
        self.language = language; self.quant = quant
        self.diarize = diarize; self.contextHint = contextHint
        self.chunk = chunk
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `swift test --filter chunkConfigDefaults && swift test --filter transcribeOptionsCarriesChunkConfig`
Expected: PASS (both).

- [ ] **Step 6: Commit**

```bash
git add Sources/ReplikaCore/ChunkConfig.swift Sources/ReplikaCore/TranscriptionProvider.swift Tests/ReplikaCoreTests/ChunkConfigTests.swift
git commit -m "feat(core): ChunkConfig + VadTier; carry chunk config in TranscribeOptions"
```

---

### Task 2: ChunkPlanner

**Files:**
- Create: `Sources/ReplikaCore/ChunkPlanner.swift`
- Test: `Tests/ReplikaCoreTests/ChunkPlannerTests.swift` (create)

**Interfaces:**
- Consumes: `ChunkConfig` (Task 1).
- Produces: `struct ChunkWindow: Sendable, Equatable` with `let start: Double`, `let end: Double`, `let overlapsPrevious: Bool` and `init(start:end:overlapsPrevious:)`;
  `enum ChunkPlanner { static func plan(spans: [(start: Double, end: Double)], config: ChunkConfig) -> [ChunkWindow] }`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/ReplikaCoreTests/ChunkPlannerTests.swift`:

```swift
import Testing
@testable import ReplikaCore

private let cfg = ChunkConfig(maxChunkSeconds: 10.0, overlapSeconds: 0.5)

@Test func emptySpansYieldNoWindows() {
    #expect(ChunkPlanner.plan(spans: [], config: cfg).isEmpty)
}

@Test func shortSpanPassesThroughUnsplit() {
    let w = ChunkPlanner.plan(spans: [(1.0, 6.0)], config: cfg)
    #expect(w.count == 1)
    #expect(w[0] == ChunkWindow(start: 1.0, end: 6.0, overlapsPrevious: false))
}

@Test func spanExactlyMaxIsNotSplit() {
    let w = ChunkPlanner.plan(spans: [(0.0, 10.0)], config: cfg)
    #expect(w.count == 1)
    #expect(w[0].overlapsPrevious == false)
    #expect(w[0].end == 10.0)
}

@Test func spanSlightlyOverMaxSplitsIntoTwoWithOverlap() {
    let w = ChunkPlanner.plan(spans: [(0.0, 10.4)], config: cfg)
    #expect(w.count == 2)
    #expect(w[0] == ChunkWindow(start: 0.0, end: 10.0, overlapsPrevious: false))
    // Second window starts overlap before the first window's end.
    #expect(w[1].start == 9.5)
    #expect(w[1].end == 10.4)
    #expect(w[1].overlapsPrevious == true)
}

@Test func longSpanSplitsIntoManyBoundedWindows() {
    let w = ChunkPlanner.plan(spans: [(0.0, 25.0)], config: cfg)
    // Windows are each <= maxChunkSeconds.
    #expect(w.allSatisfy { $0.end - $0.start <= 10.0 + 1e-9 })
    // First covers the head, later windows overlap.
    #expect(w.first?.overlapsPrevious == false)
    #expect(w.dropFirst().allSatisfy { $0.overlapsPrevious })
    // Coverage reaches the span end.
    #expect(w.last?.end == 25.0)
}

@Test func multipleSpansStayIndependentNoCrossOverlap() {
    let w = ChunkPlanner.plan(spans: [(0.0, 3.0), (5.0, 7.0)], config: cfg)
    #expect(w.count == 2)
    #expect(w[0] == ChunkWindow(start: 0.0, end: 3.0, overlapsPrevious: false))
    // A new span never overlaps the previous span (silence separates them).
    #expect(w[1] == ChunkWindow(start: 5.0, end: 7.0, overlapsPrevious: false))
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter ChunkPlanner`
Expected: FAILS TO COMPILE — `cannot find 'ChunkPlanner' in scope`.

- [ ] **Step 3: Create ChunkPlanner.swift**

Create `Sources/ReplikaCore/ChunkPlanner.swift`:

```swift
import Foundation

/// A bounded transcription window in absolute seconds. `overlapsPrevious` is
/// true only for a force-split sub-window whose start was pulled back to
/// overlap the previous sub-window of the SAME speech span.
public struct ChunkWindow: Sendable, Equatable {
    public let start: Double
    public let end: Double
    public let overlapsPrevious: Bool
    public init(start: Double, end: Double, overlapsPrevious: Bool) {
        self.start = start; self.end = end; self.overlapsPrevious = overlapsPrevious
    }
}

/// Turns VAD speech spans into transcription windows, force-splitting any span
/// longer than `config.maxChunkSeconds` into overlapping sub-windows.
public enum ChunkPlanner {
    public static func plan(spans: [(start: Double, end: Double)],
                            config: ChunkConfig) -> [ChunkWindow] {
        var windows: [ChunkWindow] = []
        for span in spans {
            if span.end - span.start <= config.maxChunkSeconds {
                windows.append(ChunkWindow(start: span.start, end: span.end,
                                           overlapsPrevious: false))
                continue
            }
            var winStart = span.start
            while winStart < span.end {
                let winEnd = min(winStart + config.maxChunkSeconds, span.end)
                windows.append(ChunkWindow(start: winStart, end: winEnd,
                                           overlapsPrevious: winStart > span.start))
                if winEnd >= span.end { break }
                winStart = winEnd - config.overlapSeconds
            }
        }
        return windows
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter ChunkPlanner`
Expected: PASS (all six).

- [ ] **Step 5: Commit**

```bash
git add Sources/ReplikaCore/ChunkPlanner.swift Tests/ReplikaCoreTests/ChunkPlannerTests.swift
git commit -m "feat(core): ChunkPlanner turns VAD spans into bounded windows with force-split overlap"
```

---

### Task 3: WordStitcher

**Files:**
- Create: `Sources/ReplikaCore/WordStitcher.swift`
- Test: `Tests/ReplikaCoreTests/WordStitcherTests.swift` (create)

**Interfaces:**
- Consumes: `Word` (existing `TranscriptTypes.swift`), `ChunkWindow` (Task 2).
- Produces: `enum WordStitcher { static func stitch(perWindow: [(window: ChunkWindow, words: [Word])]) -> [Word] }`.
  Input words are ALREADY offset to absolute time by the caller; the stitcher concatenates and de-duplicates words in the overlap region of a `overlapsPrevious` window (drops words whose midpoint is before the previous window's end).

- [ ] **Step 1: Write the failing tests**

Create `Tests/ReplikaCoreTests/WordStitcherTests.swift`:

```swift
import Testing
@testable import ReplikaCore

private func w(_ t: String, _ s: Double, _ e: Double) -> Word { Word(text: t, start: s, end: e) }

@Test func concatenatesNonOverlappingWindows() {
    let a = ChunkWindow(start: 0.0, end: 3.0, overlapsPrevious: false)
    let b = ChunkWindow(start: 5.0, end: 7.0, overlapsPrevious: false)
    let out = WordStitcher.stitch(perWindow: [
        (a, [w("hi", 0.1, 0.4), w("there", 0.5, 0.9)]),
        (b, [w("bye", 5.1, 5.4)])
    ])
    #expect(out.map(\.text) == ["hi", "there", "bye"])
}

@Test func dropsDuplicatesInForceSplitOverlap() {
    // Window B overlaps A: A ends at 10.0, B starts at 9.5. The word at ~9.6
    // (midpoint 9.65 < 10.0) is a duplicate already emitted by A → dropped.
    let a = ChunkWindow(start: 0.0, end: 10.0, overlapsPrevious: false)
    let b = ChunkWindow(start: 9.5, end: 12.0, overlapsPrevious: true)
    let out = WordStitcher.stitch(perWindow: [
        (a, [w("alpha", 9.4, 9.9)]),
        (b, [w("alpha", 9.5, 9.8), w("beta", 10.5, 10.9)])
    ])
    #expect(out.map(\.text) == ["alpha", "beta"])
}

@Test func outputIsMonotonicByStart() {
    let a = ChunkWindow(start: 0.0, end: 10.0, overlapsPrevious: false)
    let b = ChunkWindow(start: 9.5, end: 12.0, overlapsPrevious: true)
    let out = WordStitcher.stitch(perWindow: [
        (a, [w("a", 1.0, 1.2), w("b", 9.6, 9.9)]),
        (b, [w("b", 9.6, 9.9), w("c", 10.5, 10.9)])
    ])
    let starts = out.map(\.start)
    #expect(starts == starts.sorted())
}

@Test func handlesEmptyWindowsInMiddle() {
    let a = ChunkWindow(start: 0.0, end: 3.0, overlapsPrevious: false)
    let b = ChunkWindow(start: 4.0, end: 6.0, overlapsPrevious: false)
    let c = ChunkWindow(start: 7.0, end: 9.0, overlapsPrevious: false)
    let out = WordStitcher.stitch(perWindow: [
        (a, [w("a", 0.1, 0.4)]),
        (b, []),
        (c, [w("c", 7.1, 7.4)])
    ])
    #expect(out.map(\.text) == ["a", "c"])
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `swift test --filter WordStitcher`
Expected: FAILS TO COMPILE — `cannot find 'WordStitcher' in scope`.

- [ ] **Step 3: Create WordStitcher.swift**

Create `Sources/ReplikaCore/WordStitcher.swift`:

```swift
import Foundation

/// Concatenates per-window aligned words (already in absolute time) into one
/// monotonic list, de-duplicating words that fall inside the overlap region
/// shared with the previous window at a force-split boundary.
public enum WordStitcher {
    public static func stitch(perWindow: [(window: ChunkWindow, words: [Word])]) -> [Word] {
        var result: [Word] = []
        var previousEnd: Double?
        for (window, words) in perWindow {
            if window.overlapsPrevious, let boundary = previousEnd {
                for word in words where (word.start + word.end) / 2 >= boundary {
                    result.append(word)
                }
            } else {
                result.append(contentsOf: words)
            }
            previousEnd = window.end
        }
        return result
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter WordStitcher`
Expected: PASS (all four).

- [ ] **Step 5: Commit**

```bash
git add Sources/ReplikaCore/WordStitcher.swift Tests/ReplikaCoreTests/WordStitcherTests.swift
git commit -m "feat(core): WordStitcher merges per-window words, de-dups force-split overlap"
```

---

### Task 4: AudioLoader error-surface hardening

**Files:**
- Modify: `Sources/ReplikaCore/AudioLoader.swift:3-6,41,45`
- Test: `Tests/ReplikaCoreTests/AudioLoaderTests.swift` (append)

**Interfaces:**
- Produces: `AudioLoaderError` gains `case readFailed(URL)`. `loadMono16k` now maps a failed `file.read` to `.readFailed` and guards the `Int64 → UInt32` frame-count cast (mapping overflow to `.conversionFailed`) instead of trapping.

- [ ] **Step 1: Write the failing tests (append to AudioLoaderTests.swift)**

Append to `Tests/ReplikaCoreTests/AudioLoaderTests.swift`:

```swift
@Test func throwsAudioLoaderErrorOnMissingFile() {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("missing_\(UUID().uuidString).wav")
    #expect(throws: AudioLoaderError.self) {
        _ = try AudioLoader.loadMono16k(url)
    }
}

@Test func throwsAudioLoaderErrorOnCorruptFile() throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("garbage_\(UUID().uuidString).wav")
    try Data([0x00, 0x01, 0x02, 0x03, 0x04, 0x05]).write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(throws: AudioLoaderError.self) {
        _ = try AudioLoader.loadMono16k(url)
    }
}
```

- [ ] **Step 2: Run tests to verify they fail (or already error un-mapped)**

Run: `swift test --filter throwsAudioLoaderError`
Expected: BUILD OK, tests currently PASS for the missing-file case (already `.cannotOpen`) but the goal of this task is the read/overflow hardening below — proceed to make the source changes and keep these tests green. (If the corrupt-file case surfaces a raw non-`AudioLoaderError`, it FAILS here; that is exactly the escape we are closing.)

- [ ] **Step 3: Harden AudioLoader.swift**

In `Sources/ReplikaCore/AudioLoader.swift`, replace the error enum (lines 3-6):

```swift
public enum AudioLoaderError: Error {
    case cannotOpen(URL)
    case conversionFailed
    case readFailed(URL)
}
```

Replace the frame-count line (currently line 41) and the read line (currently line 45). The block from `let src = file.processingFormat` through `try file.read(into: srcBuf)` becomes:

```swift
        let src = file.processingFormat
        guard file.length >= 0,
              file.length <= AVAudioFramePosition(AVAudioFrameCount.max) else {
            throw AudioLoaderError.conversionFailed
        }
        let frames = AVAudioFrameCount(file.length)
        guard let srcBuf = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: frames) else {
            throw AudioLoaderError.conversionFailed
        }
        do { try file.read(into: srcBuf) }
        catch { throw AudioLoaderError.readFailed(url) }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `swift test --filter throwsAudioLoaderError && swift test --filter loadsAndResamplesToMono16k`
Expected: PASS (all three — the two new error-path tests and the existing happy-path test).

- [ ] **Step 5: Commit**

```bash
git add Sources/ReplikaCore/AudioLoader.swift Tests/ReplikaCoreTests/AudioLoaderTests.swift
git commit -m "fix(core): map file.read failure + guard frame-count cast in AudioLoader"
```

---

### Task 5: Provider VAD-chunk integration (loop + language plumbing + cache relocation)

**Files:**
- Modify: `Package.swift:12-19` — add the `AudioCommon` product to the `SpeechSwiftProvider` target deps (needed for `HuggingFaceDownloader`).
- Modify: `Sources/SpeechSwiftProvider/SpeechSwiftProvider.swift:1-5,86-142` (imports + `transcribe` body; add a cache-dir helper)
- Test: `Tests/SpeechSwiftProviderTests/SmokeTests.swift` (append long-form E2E)

**Interfaces:**
- Consumes: `ChunkConfig`, `ChunkPlanner.plan`, `ChunkWindow`, `WordStitcher.stitch` (Tasks 1-3), `SegmentBuilder.build`, `SpeakerMerger.merge` (existing), and the verified `speech-swift` signatures in Global Constraints.
- Produces: a `transcribe` that emits the same `TranscriptEvent` stream as before, now correct on long audio; a `static func modelCacheDir(for:) throws -> URL` on `SpeechSwiftProvider`.

- [ ] **Step 1: Write the failing E2E regression test (append to SmokeTests.swift)**

Append to `Tests/SpeechSwiftProviderTests/SmokeTests.swift`:

```swift
@Test(.enabled(if: smokeAudioURL() != nil))
func longFormProducesManySegments() async throws {
    let url = try #require(smokeAudioURL())
    let samples = try AudioLoader.loadMono16k(url)
    let durationSec = Double(samples.count) / 16000.0

    let provider = SpeechSwiftProvider()
    var done: Transcript?
    for try await ev in provider.transcribe(.file(url), options: TranscribeOptions(diarize: true)) {
        if case .done(let t) = ev { done = t }
    }
    let transcript = try #require(done)

    // Regression on FINDINGS §5: single-shot transcribe() returned 0 segments
    // on the ~9.4 min clip. Chunking must yield many labeled segments.
    #expect(transcript.segments.count >= 5)

    // Cross-segment monotonicity and coverage reaching well into the clip.
    let starts = transcript.segments.map(\.start)
    #expect(starts == starts.sorted())
    if let last = transcript.segments.last {
        #expect(last.end > durationSec * 0.5)
    }

    // Metrics only — never transcript text.
    let coverage = (transcript.segments.last?.end ?? 0) / durationSec * 100
    print("LONGFORM: segments=\(transcript.segments.count), coverage=\(String(format: "%.0f%%", coverage))")
}
```

- [ ] **Step 2: Confirm it fails against the current single-shot provider**

Run (with env pointing at the FULL clip):
```bash
RUN_MODEL_TESTS=1 SMOKE_AUDIO="$PWD/call_2026-07-14_15-05-51.m4a" swift test --filter longFormProducesManySegments
```
Expected: FAIL — `transcript.segments.count >= 5` is false (current provider yields 0 segments on long audio). If `SMOKE_AUDIO`/`RUN_MODEL_TESTS` are unset the test is skipped; set them to see the red.

- [ ] **Step 3: Add the AudioCommon dependency, then rewrite the provider imports and `transcribe` body**

First, in `Package.swift`, add the `AudioCommon` product to the `SpeechSwiftProvider` target's `dependencies` (it currently lists `ReplikaCore`, `Qwen3ASR`, `SpeechVAD`) so `import AudioCommon` / `HuggingFaceDownloader` resolve:

```swift
        .target(
            name: "SpeechSwiftProvider",
            dependencies: [
                "ReplikaCore",
                .product(name: "Qwen3ASR", package: "speech-swift"),
                .product(name: "SpeechVAD", package: "speech-swift"),
                .product(name: "AudioCommon", package: "speech-swift")
            ]
        ),
```

Then, in `Sources/SpeechSwiftProvider/SpeechSwiftProvider.swift`, set the imports (lines 1-5) to:

```swift
import Foundation
import os
import ReplikaCore
import Qwen3ASR
import SpeechVAD
import AudioCommon
```

Replace the entire `transcribe(_:options:)` method (currently lines 86-142) with:

```swift
    public func transcribe(_ audio: AudioSource,
                           options: TranscribeOptions) -> AsyncThrowingStream<TranscriptEvent, Error> {
        AsyncThrowingStream { continuation in
            let work = Task {
                do {
                    guard case let .file(url) = audio else { throw ProviderError.unsupportedSource }
                    let samples = try AudioLoader.loadMono16k(url)
                    continuation.yield(.progress(0.05))

                    // Diarization (optional) — runs on the full buffer.
                    var tags: [SpeakerTag] = []
                    if options.diarize {
                        let diarizer = try await SortformerDiarizer.fromPretrained(
                            cacheDir: try Self.modelCacheDir(for: SortformerDiarizer.defaultModelId))
                        try Task.checkCancellation()
                        let result = diarizer.diarize(audio: samples, sampleRate: 16000, config: .default)
                        self.logger.info("diarization found \(result.numSpeakers, privacy: .public) distinct speaker(s)")
                        tags = result.segments.map {
                            SpeakerTag(speaker: $0.speakerId, start: Double($0.startTime), end: Double($0.endTime))
                        }
                        for tag in tags { continuation.yield(.speaker(tag)) }
                    }
                    continuation.yield(.progress(0.3))

                    // Load VAD + ASR + aligner (cache relocated to Application Support).
                    let vad = try await SileroVADModel.fromPretrained(
                        cacheDir: try Self.modelCacheDir(for: SileroVADModel.defaultModelId))
                    let model = try await Qwen3ASRModel.fromPretrained(
                        modelId: options.quant.asrModelId,
                        cacheDir: try Self.modelCacheDir(for: options.quant.asrModelId))
                    let aligner = try await Qwen3ForcedAligner.fromPretrained(
                        modelId: options.quant.alignerModelId,
                        cacheDir: try Self.modelCacheDir(for: options.quant.alignerModelId))
                    try Task.checkCancellation()

                    // VAD → bounded windows.
                    let vadConfig = VADConfig(
                        onset: VADConfig.sileroDefault.onset,
                        offset: VADConfig.sileroDefault.offset,
                        minSpeechDuration: Float(options.chunk.minSpeechSeconds),
                        minSilenceDuration: VADConfig.sileroDefault.minSilenceDuration,
                        windowDuration: VADConfig.sileroDefault.windowDuration,
                        stepRatio: VADConfig.sileroDefault.stepRatio)
                    let spans = vad.detectSpeech(audio: samples, sampleRate: 16000, config: vadConfig)
                        .map { (start: Double($0.startTime), end: Double($0.endTime)) }
                    let windows = ChunkPlanner.plan(spans: spans, config: options.chunk)

                    // `language` here is a word-splitting hint threaded from the
                    // request (removes the hardcoded "English"); RU uses the
                    // whitespace fallback either way, CJK benefits from the real value.
                    let lang: String? = options.language == "auto" ? nil : options.language

                    // Per-window transcribe + align, offset to absolute time.
                    var perWindow: [(window: ChunkWindow, words: [Word])] = []
                    let total = max(windows.count, 1)
                    for (i, window) in windows.enumerated() {
                        try Task.checkCancellation()
                        let startSample = Int(window.start * 16000)
                        let endSample = min(Int(window.end * 16000), samples.count)
                        guard startSample < endSample else {
                            perWindow.append((window, []))
                            continue
                        }
                        let spanAudio = Array(samples[startSample..<endSample])
                        let text = model.transcribe(audio: spanAudio, sampleRate: 16000,
                                                    language: lang, context: options.contextHint)
                        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                        if trimmed.isEmpty {
                            perWindow.append((window, []))
                        } else {
                            let aligned = aligner.align(audio: spanAudio, text: text,
                                                        sampleRate: 16000, language: lang ?? "English")
                            let offset = window.start
                            let words = aligned.map {
                                Word(text: $0.text,
                                     start: Double($0.startTime) + offset,
                                     end: Double($0.endTime) + offset)
                            }
                            perWindow.append((window, words))
                        }
                        continuation.yield(.progress(0.3 + 0.6 * Double(i + 1) / Double(total)))
                    }

                    // Stitch → segments → speakers (downstream unchanged).
                    let globalWords = WordStitcher.stitch(perWindow: perWindow)
                    let base = SegmentBuilder.build(words: globalWords)
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

    /// Per-model cache directory under the app's Application Support, replacing
    /// speech-swift's default `~/Library/Caches/qwen3-speech/`. Reuses upstream's
    /// Hub-style per-model layout via `getCacheDirectory(for:basePath:)`, so each
    /// model still lands in its own subdirectory (no collisions).
    static func modelCacheDir(for modelId: String) throws -> URL {
        let base = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
            .appendingPathComponent("com.replika.spike", isDirectory: true)
        return try HuggingFaceDownloader.getCacheDirectory(for: modelId, basePath: base)
    }
```

- [ ] **Step 4: Build to verify it compiles warning-free**

Run: `swift build --target SpeechSwiftProvider`
Expected: Compiles with no warnings. If `SileroVADModel.defaultModelId` / `SortformerDiarizer.defaultModelId` resolve, imports are correct.

- [ ] **Step 5: Run the long-form E2E to verify it passes**

Run (FULL clip):
```bash
RUN_MODEL_TESTS=1 SMOKE_AUDIO="$PWD/call_2026-07-14_15-05-51.m4a" swift test --filter longFormProducesManySegments
```
Expected: PASS. Console shows `LONGFORM: segments=<N>, coverage=<~high>%` with N ≥ 5. Also confirm the existing `smokeProducesLabeledSegments` still passes (short-clip regression):
```bash
RUN_MODEL_TESTS=1 SMOKE_AUDIO="$PWD/fixtures/call_60s.wav" swift test --filter smokeProducesLabeledSegments
```
Expected: PASS.

NOTE (spec §11 risk): read the printed distinct-speaker count from `smokeProducesLabeledSegments` and eyeball speaker sanity on the long clip. If diarization returns an implausible speaker count on the ~9.4 min buffer, record it for a follow-up (chunk diarization) — it does not block this task's segment-count goal but must be surfaced in Task 7.

- [ ] **Step 6: Commit**

```bash
git add Package.swift Sources/SpeechSwiftProvider/SpeechSwiftProvider.swift Tests/SpeechSwiftProviderTests/SmokeTests.swift
git commit -m "feat(provider): VAD-chunked long-form transcribe; thread language; relocate model cache"
```

---

### Task 6: Bench extension (segment count)

**Files:**
- Modify: `Sources/replika-spike/Bench.swift:16-31`
- Modify: `Sources/replika-spike/main.swift:66-88`

**Interfaces:**
- Consumes: `BenchResult` (existing).
- Produces: `BenchResult` gains `let segments: Int`; its markdown row/header include a `segments` column. `main.swift` counts `.committed` events and passes the count.

- [ ] **Step 1: Extend BenchResult**

In `Sources/replika-spike/Bench.swift`, replace the `BenchResult` struct (lines 16-31) with:

```swift
struct BenchResult {
    let clip: String
    let quant: String
    let rtf: Double
    let peakMB: Double
    let loadAndRunSec: Double
    let segments: Int

    var markdownRow: String {
        String(format: "| %@ | %@ | %.2f | %.0f | %.1f | %d |",
               clip, quant, rtf, peakMB, loadAndRunSec, segments)
    }

    static var markdownHeader: String {
        "| clip | quant | RTF | peak RAM (MB) | wall (s) | segments |\n|---|---|---|---|---|---|"
    }
}
```

- [ ] **Step 2: Count segments in main.swift**

In `Sources/replika-spike/main.swift`, add a counter before the stream loop (just after line 64 `let options = ...`):

```swift
var segmentCount = 0
```

In the `for try await ev` loop, change the `.committed` case to also count:

```swift
    case .committed(let seg):
        segmentCount += 1
        let who = seg.speaker.map { "S\($0)" } ?? "S?"
        print("[\(fmt(seg.start))–\(fmt(seg.end))] \(who): \(seg.text)")
```

And add `segments: segmentCount` to the `BenchResult(...)` initializer:

```swift
let result = BenchResult(
    clip: clipName,
    quant: quant.rawValue,
    rtf: audioSeconds / max(wall, 0.0001),
    peakMB: Double(peak.value) / 1_048_576.0,
    loadAndRunSec: wall,
    segments: segmentCount
)
```

- [ ] **Step 3: Build to verify it compiles**

Run: `swift build --target replika-spike`
Expected: Compiles with no errors.

- [ ] **Step 4: Smoke-run the CLI on the 60 s fixture**

Run: `swift run replika-spike fixtures/call_60s.wav --quant q4`
Expected: prints per-segment lines and a final markdown table whose header ends with `| segments |` and whose row ends with a non-zero segment count.

- [ ] **Step 5: Commit**

```bash
git add Sources/replika-spike/Bench.swift Sources/replika-spike/main.swift
git commit -m "feat(cli): add segment-count column to bench output"
```

---

### Task 7: Final verification + FINDINGS/progress update

**Files:**
- Modify: `docs/superpowers/specs/2026-07-15-mlx-asr-feasibility-spike-FINDINGS.md` (bench table row + §5 status)
- Modify: `.superpowers/sdd/progress.md` (append slice-B result note)

**Interfaces:** none (verification + docs).

- [ ] **Step 1: Full unit suite (no models)**

Run: `swift test`
Expected: All non-model tests PASS; model-gated tests SKIPPED (no `RUN_MODEL_TESTS`). Zero failures, zero warnings.

- [ ] **Step 2: Run the bench on the full clip, both quant tiers**

Run:
```bash
swift run replika-spike call_2026-07-14_15-05-51.m4a --quant q4
swift run replika-spike call_2026-07-14_15-05-51.m4a --quant q8
```
Expected: BOTH now emit a non-zero segment count (the old result was 0). Capture RTF / peak RAM / wall / segments from each markdown row. Sanity-check the printed speaker labels reflect a 2-party call.

- [ ] **Step 3: Update FINDINGS bench table**

In `docs/superpowers/specs/2026-07-15-mlx-asr-feasibility-spike-FINDINGS.md`, update the §1 bench table rows for the 9.4-min clip with the real numbers from Step 2 (replace the `**0**` segments / "не определено" cells), and add a one-line note under §5 that chunking (slice B) closed the blocker, linking the slice-B design + plan. Keep the privacy rule: metrics only, no transcript text.

- [ ] **Step 4: Note the diarization-on-long-buffer finding**

Record in the same FINDINGS note (and `.superpowers/sdd/progress.md`) whether Sortformer diarization returned a plausible speaker count on the full clip (spec §11 risk). If it degraded, add a follow-up bullet "chunk diarization" to the deferred-gaps list; if it held up, state that explicitly.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-15-mlx-asr-feasibility-spike-FINDINGS.md .superpowers/sdd/progress.md
git commit -m "docs: close long-form blocker in FINDINGS; record slice-B bench + diarization check"
```

---

## Self-Review

**1. Spec coverage:**
- §1 blocker → Tasks 2,3,5 (chunking spine). ✅
- §2 API reconciliation → Global Constraints (verified signatures). ✅
- §3 scope: chunking (T1-3,5), aligner-language (T5), AudioLoader-hardening (T4), cache-relocation (T5). Deferred items not implemented. ✅
- §4 architecture split (ReplikaCore pure vs provider) → T1-3 pure, T5 provider. ✅
- §5 data flow → T5 body. ✅
- §6 ChunkConfig → T1. ✅
- §7 overlap/stitch → T2 (force-split), T3 (dedup). ✅
- §8 error/progress (zero spans, empty span skip, cancellation, per-chunk progress, AudioLoader) → T4, T5. ✅
- §9 tests (ChunkPlanner, WordStitcher, AudioLoader error path, long-form E2E) → T2,T3,T4,T5. ✅
- §10 bench → T6, T7. ✅
- §11 risks (diarization-on-long-buffer, overlap-dedup, peak RAM) → T5 note, T7 Step 4, T7 Step 2. ✅

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✅

**3. Type consistency:** `ChunkConfig` fields (`maxChunkSeconds`, `overlapSeconds`, `vadTier`, `minSpeechSeconds`) identical across T1/T2/T5. `ChunkWindow(start:end:overlapsPrevious:)` identical across T2/T3/T5. `WordStitcher.stitch(perWindow:)` tuple shape `(window: ChunkWindow, words: [Word])` identical in T3/T5. `BenchResult(...)` includes `segments:` in both T6 edits. ✅
