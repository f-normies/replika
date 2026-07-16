@preconcurrency import AVFoundation

public enum AudioLoaderError: Error {
    case cannotOpen(URL)
    case conversionFailed
    case readFailed(URL)
}

/// Feeds a single already-filled `AVAudioPCMBuffer` to `AVAudioConverter.convert`
/// exactly once. `AVAudioConverter.convert`'s input block is `@Sendable` under
/// strict concurrency, but the callback in fact runs synchronously within the
/// `convert(...)` call frame on the calling thread (no concurrent/async
/// invocation ever occurs), so wrapping the one-shot state in an
/// `@unchecked Sendable` class is safe here rather than racy.
private final class InputProvider: @unchecked Sendable {
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

public enum AudioLoader {
    /// Decode any AVFoundation-supported file to 16 kHz mono Float32 samples.
    public static func loadMono16k(_ url: URL) throws -> [Float] {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) }
        catch { throw AudioLoaderError.cannotOpen(url) }

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

        guard let dst = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                      channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: src, to: dst) else {
            throw AudioLoaderError.conversionFailed
        }

        let cap = AVAudioFrameCount(Double(frames) * 16000.0 / src.sampleRate) + 1024
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: dst, frameCapacity: cap) else {
            throw AudioLoaderError.conversionFailed
        }

        let provider = InputProvider(srcBuf)
        var convError: NSError?
        let status = conv.convert(to: dstBuf, error: &convError) { _, outStatus in
            provider.next(outStatus)
        }
        if status == .error || convError != nil {
            throw AudioLoaderError.conversionFailed
        }

        guard let ch = dstBuf.floatChannelData else { throw AudioLoaderError.conversionFailed }
        return Array(UnsafeBufferPointer(start: ch[0], count: Int(dstBuf.frameLength)))
    }
}
