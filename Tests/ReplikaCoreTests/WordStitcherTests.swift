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
