import Foundation
@preconcurrency import AVFoundation
import Qwen3ASR

// Reconciled speech-swift API (verified against .build/checkouts/speech-swift
// @ 335a68c0db92564527881962ff11eb686863d2b8, Sources/Qwen3ASR/*.swift):
//
//   public extension Qwen3ASRModel {
//       static func fromPretrained(
//           modelId: String = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
//           cacheDir: URL? = nil,
//           offlineMode: Bool = false,
//           progressHandler: ((Double, String) -> Void)? = nil
//       ) async throws -> Qwen3ASRModel
//   }
//   extension Qwen3ASRModel {
//       func transcribe(
//           audio: [Float], sampleRate: Int = 16000,
//           language: String? = nil, maxTokens: Int = 448, context: String? = nil
//       ) -> String                               // NOT async, NOT throws
//   }
//
//   public extension Qwen3ForcedAligner {
//       static func fromPretrained(
//           modelId: String = "aufklarer/Qwen3-ForcedAligner-0.6B-4bit",
//           cacheDir: URL? = nil,
//           offlineMode: Bool = false,
//           progressHandler: ((Double, String) -> Void)? = nil
//       ) async throws -> Qwen3ForcedAligner
//   }
//   extension Qwen3ForcedAligner {
//       func align(
//           audio: [Float], text: String,
//           sampleRate: Int = 16000, language: String = "English"
//       ) -> [AlignedWord]                         // NOT async, NOT throws
//   }
//
//   public struct AlignedWord (defined in AudioCommon/Protocols.swift,
//   re-exported through Qwen3ASR): .text: String, .startTime: Float, .endTime: Float
//
// Delta vs. the brief's assumed signatures: all core shapes (async/throws-ness,
// property names) matched exactly. The only difference is that fromPretrained
// on both types takes extra *optional* parameters (modelId, cacheDir,
// offlineMode, progressHandler) with defaults, so calling with `()` still works.
// `align`'s `language` parameter is non-optional with default "English"; for
// Russian text this is harmless because non-whitespace-delimited scripts
// (Japanese/Korean/Thai/etc.) are special-cased and everything else
// (including Russian) falls through to the same whitespace-splitting path.
//
// Quantization selection mechanism (secondary discovery): there is no
// separate quant/variant enum argument. `fromPretrained(modelId:)` takes a
// HuggingFace repo-id string, and the bits/size are inferred by substring
// matching on that string (see `ASRModelSize.detect`/`.detectBits` in
// Qwen3ASR.swift): "1.7B"/"1.7b" -> large, else small; "8bit"/"8-bit" -> 8,
// "5bit"/"5-bit" -> 5, "4bit"/"4-bit" -> 4, else default-by-size (4 for
// small, 8 for large). Default model id is
// "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"; switching quant/size means passing a
// different repo-id string, e.g. "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
// (Qwen3ASRModel.largeModelId) or "aufklarer/Qwen3-ASR-0.6B-MLX-8bit"
// (mentioned in the in-package RAM-warning message, not a named constant).

// Feeds a single already-filled `AVAudioPCMBuffer` to `AVAudioConverter.convert`
// exactly once. `AVAudioConverter.convert`'s input block is `@Sendable` under
// strict concurrency, but the callback in fact runs synchronously within the
// `convert(...)` call frame on the calling thread (no concurrent/async
// invocation ever occurs), so wrapping the one-shot state in an
// `@unchecked Sendable` class is safe here rather than racy.
final class InputProvider: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private var provided = false

    init(_ buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(_ status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioPCMBuffer? {
        if provided {
            status.pointee = .noDataNow
            return nil
        }
        provided = true
        status.pointee = .haveData
        return buffer
    }
}

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
    let provider = InputProvider(srcBuf)
    _ = conv.convert(to: dstBuf, error: nil) { _, status in provider.next(status) }
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
