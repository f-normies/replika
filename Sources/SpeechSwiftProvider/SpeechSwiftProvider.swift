import Foundation
import os
import ReplikaCore
import Qwen3ASR
import SpeechVAD
import AudioCommon

/// Errors specific to `SpeechSwiftProvider`.
public enum ProviderError: Error {
    case unsupportedSource
}

// Quant-tier -> HuggingFace model-id mapping for the ASR + forced-aligner
// backends. `speech-swift` has no dedicated `quantBits:`/`variant:` argument;
// `fromPretrained(modelId:)` instead auto-detects size/bits by substring
// matching on the model-id string itself (`ASRModelSize.detect`/`.detectBits`
// in Qwen3ASR.swift; `ForcedAlignerVariant.detect` in ForcedAligner.swift —
// reconciled against source in Task 2's report).
//
// ASR model size: pinned to Qwen3-ASR-**1.7B** per the spec/architecture
// default (acceptance criterion 1) — both quant tiers below select the 1.7B
// weights, only the quantization bit-width changes, so a q4 vs. q8 bench
// comparison isolates the quant effect.
//
// Forced-aligner model size: stays **0.6B** for both tiers — the aligner has
// no 1.7B variant upstream (only `ForcedAlignerVariant.mlx4bit`/`.mlx8bit`,
// both 0.6B), so this is an unavoidable size mismatch with the ASR model,
// not an oversight.
private extension Quant {
    /// Argument for `Qwen3ASRModel.fromPretrained(modelId:)`.
    var asrModelId: String {
        switch self {
        // "aufklarer/Qwen3-ASR-1.7B-MLX-4bit": referenced only in an
        // in-package RAM-warning string (Qwen3ASR.swift:1190), not exposed
        // as a named constant upstream (unlike `.largeModelId` below).
        case .q4: return "aufklarer/Qwen3-ASR-1.7B-MLX-4bit"
        case .q8: return Qwen3ASRModel.largeModelId
        }
    }

    /// Argument for `Qwen3ForcedAligner.fromPretrained(modelId:)`.
    var alignerModelId: String {
        switch self {
        case .q4: return ForcedAlignerVariant.mlx4bit.rawValue
        case .q8: return ForcedAlignerVariant.mlx8bit.rawValue
        }
    }
}

/// Wraps `speech-swift`'s Qwen3-ASR + forced aligner + Sortformer diarizer
/// behind the `TranscriptionProvider` protocol.
///
/// Diarization uses `SortformerDiarizer` (`SpeechVAD`, CoreML-backed,
/// end-to-end neural diarization) — reconciled against source because the
/// originally-assumed `DiarizationPipeline.fromPretrained()` /
/// `.diarize(audio:sampleRate:)` shape returning `.speakerId`/`.startTime`/
/// `.endTime` directly does not exist as such. What exists:
///   - `DiarizationPipeline` is a `typealias` for `PyannoteDiarizationPipeline`
///     (embedding + clustering based, not Sortformer).
///   - `SortformerDiarizer.fromPretrained(modelId:cacheDir:offlineMode:config:
///     computeUnits:progressHandler:) async throws -> SortformerDiarizer`
///     (`defaultModelId = "aufklarer/Sortformer-Diarization-CoreML"`).
///   - `SortformerDiarizer.diarize(audio:sampleRate:config:) -> DiarizationResult`
///     where `DiarizationResult` has `.segments: [DiarizedSegment]`,
///     `.numSpeakers: Int`, `.speakerEmbeddings: [[Float]]`.
///   - `DiarizedSegment` has `.startTime: Float`, `.endTime: Float`,
///     `.speakerId: Int` (all `Float`, not `Double` — needs an explicit
///     conversion into `SpeakerTag`, which stores `Double`).
///
/// - Important: Concurrency contract — this provider supports **one active
///   transcription per instance**. `transcribe(_:options:)` stores its
///   in-flight `Task` in `self.task` under a lock; calling it again before
///   the previous stream finishes overwrites the tracked task with the new
///   one. `cancel()` only ever cancels whatever `self.task` currently holds,
///   so it reaches the latest call, not any earlier still-running one. Run
///   concurrent transcriptions from separate `SpeechSwiftProvider` instances
///   if independent cancellation is required.
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

    public func cancel() {
        lock.lock(); let t = task; lock.unlock()
        t?.cancel()
    }
}
