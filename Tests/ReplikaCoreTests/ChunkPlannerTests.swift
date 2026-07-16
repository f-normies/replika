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

@Test func degenerateOverlapTerminatesWithBoundedWindows() {
    // overlapSeconds >= maxChunkSeconds must not hang; windows stay bounded.
    let bad = ChunkConfig(maxChunkSeconds: 5.0, overlapSeconds: 5.0)
    let w = ChunkPlanner.plan(spans: [(0.0, 12.0)], config: bad)
    #expect(!w.isEmpty)
    #expect(w.count < 100)
    #expect(w.allSatisfy { $0.end - $0.start <= 5.0 + 1e-9 })
    #expect(w.last?.end == 12.0)
}

@Test func nonPositiveMaxYieldsSingleWindowPerSpan() {
    let bad = ChunkConfig(maxChunkSeconds: 0.0, overlapSeconds: 0.0)
    let w = ChunkPlanner.plan(spans: [(0.0, 12.0)], config: bad)
    #expect(w.count == 1)
    #expect(w[0] == ChunkWindow(start: 0.0, end: 12.0, overlapsPrevious: false))
}
