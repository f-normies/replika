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

@Test func chainedOverlapUsesImmediatelyPreviousWindowEnd() {
    // A (non-overlap) end=10; B (overlap) boundary must be A.end=10;
    // C (overlap) boundary must be B.end=19.5, NOT A.end. A word at ~19.3
    // (mid 19.3) is a duplicate of B's tail: dropped only if C uses B.end.
    let a = ChunkWindow(start: 0.0, end: 10.0, overlapsPrevious: false)
    let b = ChunkWindow(start: 9.5, end: 19.5, overlapsPrevious: true)
    let c = ChunkWindow(start: 19.0, end: 29.0, overlapsPrevious: true)
    let out = WordStitcher.stitch(perWindow: [
        (a, [w("a", 1.0, 1.2)]),
        (b, [w("dupA", 9.6, 9.9), w("b", 11.0, 11.2)]),   // dupA mid 9.75 < 10 dropped
        (c, [w("dupB", 19.2, 19.4), w("c", 20.0, 20.2)])  // dupB mid 19.3 < 19.5 dropped
    ])
    // If C wrongly used A.end (10) as its boundary, dupB (mid 19.3 >= 10) would survive.
    #expect(out.map(\.text) == ["a", "b", "c"])
}

@Test func keepsWordExactlyAtBoundary() {
    // Midpoint exactly equal to the boundary is kept (>=), not dropped.
    let a = ChunkWindow(start: 0.0, end: 10.0, overlapsPrevious: false)
    let b = ChunkWindow(start: 9.5, end: 15.0, overlapsPrevious: true)
    let out = WordStitcher.stitch(perWindow: [
        (a, []),
        (b, [w("tie", 9.0, 11.0)])   // midpoint (9+11)/2 == 10.0 == boundary -> kept
    ])
    #expect(out.map(\.text) == ["tie"])
}

@Test func emptyForceSplitPredecessorKeepsSuccessorOverlapWords() {
    // W0 (force-split head) transcribed empty; W1 overlaps it. W1's word in the
    // overlap band [9.5,10] must NOT be dropped — W0 emitted nothing to duplicate.
    let w0 = ChunkWindow(start: 0.0, end: 10.0, overlapsPrevious: false)
    let w1 = ChunkWindow(start: 9.5, end: 19.5, overlapsPrevious: true)
    let out = WordStitcher.stitch(perWindow: [
        (w0, []),
        (w1, [w("x", 9.6, 9.9), w("y", 11.0, 11.2)])
    ])
    #expect(out.map(\.text) == ["x", "y"])
}
