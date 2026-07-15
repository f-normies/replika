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
