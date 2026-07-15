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
