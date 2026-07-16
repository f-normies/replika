import Testing
@testable import ReplikaCore

@Test func chunkConfigDefaults() {
    let c = ChunkConfig()
    #expect(c.maxChunkSeconds == 10.0)
    #expect(c.overlapSeconds == 0.5)
    #expect(c.vadTier == .silero)
    #expect(c.minSpeechSeconds == 0.25)
}

@Test func transcribeOptionsCarriesChunkConfig() {
    let opts = TranscribeOptions(chunk: ChunkConfig(maxChunkSeconds: 5.0))
    #expect(opts.chunk.maxChunkSeconds == 5.0)
    // Default still applies when omitted.
    #expect(TranscribeOptions().chunk.maxChunkSeconds == 10.0)
}
