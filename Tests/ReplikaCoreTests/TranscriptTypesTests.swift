import Testing
import Foundation
@testable import ReplikaCore

@Test func segmentDefaultsToNoSpeaker() {
    let seg = Segment(text: "hi", start: 0, end: 1, words: [Word(text: "hi", start: 0, end: 1)])
    #expect(seg.speaker == nil)
    #expect(seg.words.count == 1)
}

@Test func transcribeOptionsDefaults() {
    let o = TranscribeOptions()
    #expect(o.language == "auto")
    #expect(o.quant == .q4)
    #expect(o.diarize == true)
}
