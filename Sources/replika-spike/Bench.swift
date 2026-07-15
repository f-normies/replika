import Foundation
import Darwin

/// Current physical memory footprint of this process, in bytes.
func currentFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size) / 4
    let kr = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

struct BenchResult {
    let clip: String
    let quant: String
    let rtf: Double
    let peakMB: Double
    let loadAndRunSec: Double

    var markdownRow: String {
        String(format: "| %@ | %@ | %.2f | %.0f | %.1f |",
               clip, quant, rtf, peakMB, loadAndRunSec)
    }

    static var markdownHeader: String {
        "| clip | quant | RTF | peak RAM (MB) | wall (s) |\n|---|---|---|---|---|"
    }
}
