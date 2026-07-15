import Testing
@testable import ReplikaCore

@Test func assignsSpeakerWithMaxOverlap() {
    let seg = Segment(text: "hello", start: 1.0, end: 3.0,
                      words: [Word(text: "hello", start: 1.0, end: 3.0)])
    let tags = [
        SpeakerTag(speaker: 0, start: 0.0, end: 1.4),  // 0.4 s overlap
        SpeakerTag(speaker: 1, start: 1.4, end: 3.0)   // 1.6 s overlap -> wins
    ]
    let merged = SpeakerMerger.merge(segments: [seg], tags: tags)
    #expect(merged[0].speaker == 1)
}

@Test func noTagsLeavesSpeakerNil() {
    let seg = Segment(text: "x", start: 0, end: 1, words: [Word(text: "x", start: 0, end: 1)])
    #expect(SpeakerMerger.merge(segments: [seg], tags: []).first?.speaker == nil)
}
