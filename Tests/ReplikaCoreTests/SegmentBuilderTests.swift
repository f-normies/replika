import Testing
@testable import ReplikaCore

@Test func splitsOnLongPause() {
    let words = [
        Word(text: "a", start: 0.0, end: 0.3),
        Word(text: "b", start: 0.35, end: 0.6),   // small gap -> same segment
        Word(text: "c", start: 2.0, end: 2.3)      // >0.6 s gap -> new segment
    ]
    let segs = SegmentBuilder.build(words: words, pauseThreshold: 0.6)
    #expect(segs.count == 2)
    #expect(segs[0].text == "a b")
    #expect(segs[1].text == "c")
    #expect(segs[0].start == 0.0)
    #expect(segs[0].end == 0.6)
}

@Test func emptyWordsYieldNoSegments() {
    #expect(SegmentBuilder.build(words: []).isEmpty)
}
