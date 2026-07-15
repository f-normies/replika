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
