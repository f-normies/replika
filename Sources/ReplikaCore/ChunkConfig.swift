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
