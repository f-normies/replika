import Testing
import Foundation
import ReplikaCore
@testable import SpeechSwiftProvider

// PRIVACY: this suite intentionally does NOT bundle any audio resource.
// The clip path comes from the `SMOKE_AUDIO` env var, and the test only
// runs when BOTH `RUN_MODEL_TESTS == "1"` AND `SMOKE_AUDIO` are set to a
// non-empty value — otherwise it is skipped, keeping the offline suite green.
private func smokeAudioURL() -> URL? {
    let env = ProcessInfo.processInfo.environment
    guard env["RUN_MODEL_TESTS"] == "1",
          let path = env["SMOKE_AUDIO"], !path.isEmpty else {
        return nil
    }
    return URL(fileURLWithPath: path)
}

@Test(.enabled(if: smokeAudioURL() != nil))
func smokeProducesLabeledSegments() async throws {
    let url = try #require(smokeAudioURL())
    let provider = SpeechSwiftProvider()

    var done: Transcript?
    var sawSpeaker = false
    var speakerIDs = Set<Int>()
    for try await ev in provider.transcribe(.file(url), options: TranscribeOptions(diarize: true)) {
        switch ev {
        case .speaker(let tag):
            sawSpeaker = true
            speakerIDs.insert(tag.speaker)
        case .done(let t): done = t
        default: break
        }
    }

    // Distinct-speaker count is a metric, not transcript content — safe to
    // surface for the spike's key result (diarization signal on the RU clip).
    print("SMOKE: distinct speakers = \(speakerIDs.count)")

    let transcript = try #require(done)
    #expect(!transcript.segments.isEmpty)
    #expect(transcript.segments.allSatisfy { !$0.text.isEmpty })
    #expect(sawSpeaker)
    for seg in transcript.segments where seg.words.count > 1 {
        for i in 1..<seg.words.count {
            #expect(seg.words[i].start >= seg.words[i - 1].start)  // monotonic
            #expect(seg.words[i].end <= seg.end + 0.001)           // within segment
        }
    }
}
