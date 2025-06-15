import sys
import math

class DNNSim:
    def __init__(self, M, K, N):
        # workload dims; assume multiples of 8
        self.M = M
        self.K = K
        self.N = N

        # matrix unit specs (8x8x8 op in 16 cycles)
        self.tile_m = 8
        self.tile_k = 8
        self.tile_n = 8
        self.cycles_per_tile = 16
        self.num_mu = 8  # 8 matrix units in parallel

        # sram specs
        self.sram_size = 2 * 1024 * 1024  # 2mb in bytes
        self.sram_read_bw = 128  # bytes/cycle (read)
        self.sram_write_bw = 64  # bytes/cycle (write)

        # dram specs
        self.dram_read_bw = 4   # bytes/cycle (read)
        self.dram_write_bw = 2  # bytes/cycle (write)

    def compute_tile_counts(self):
        # calc number of mu-level tiles; assume dims divisible by 8
        num_tiles_m = self.M // self.tile_m
        num_tiles_k = self.K // self.tile_k
        num_tiles_n = self.N // self.tile_n
        total_tiles = num_tiles_m * num_tiles_k * num_tiles_n
        return total_tiles, num_tiles_m, num_tiles_k, num_tiles_n

    def compute_compute_cycles(self, total_tiles):
        # each tile op takes fixed cycles; mu run in parallel
        cycles = math.ceil(total_tiles / self.num_mu) * self.cycles_per_tile
        return cycles

    def compute_sram_mu_transfers(self, total_tiles):
        # each mu tile op: read two 8x8 matrices (ifmap & weight) and write one output
        # each 8x8 = 64 bytes; so per tile: 64+64=128 read, 64 write
        bytes_read_per_tile = 128
        bytes_write_per_tile = 64

        total_read_bytes = total_tiles * bytes_read_per_tile
        total_write_bytes = total_tiles * bytes_write_per_tile

        # cycles based on sram bw
        # using ceil since partial cycles count as full cycle
        read_cycles = math.ceil(total_read_bytes / self.sram_read_bw)
        write_cycles = math.ceil(total_write_bytes / self.sram_write_bw)
        return (read_cycles, write_cycles, total_read_bytes, total_write_bytes)

    def compute_dram_sram_transfers(self):
        # dram transfers: load ifmap and weight, store ofmap
        # ifmap: M x K, weight: K x N, ofmap: M x N; int8 so 1 byte each
        total_dram_read_bytes = self.M * self.K + self.K * self.N
        total_dram_write_bytes = self.M * self.N

        read_cycles = math.ceil(total_dram_read_bytes / self.dram_read_bw)
        write_cycles = math.ceil(total_dram_write_bytes / self.dram_write_bw)
        return (read_cycles, write_cycles, total_dram_read_bytes, total_dram_write_bytes)

    def run_simulation(self):
        # get tile counts
        total_tiles, tm, tk, tn = self.compute_tile_counts()
        # calc compute cycles
        compute_cycles = self.compute_compute_cycles(total_tiles)
        # calc sram->mu transfers cycles and bytes
        sram_mu_read_cycles, sram_mu_write_cycles, sram_mu_read_bytes, sram_mu_write_bytes = self.compute_sram_mu_transfers(total_tiles)
        # calc dram->sram transfers cycles and bytes
        dram_read_cycles, dram_write_cycles, dram_read_bytes, dram_write_bytes = self.compute_dram_sram_transfers()

        # total execution cycles (overlap assumed) is the max of compute and all transfer cycles
        total_cycles = max(compute_cycles, sram_mu_read_cycles, sram_mu_write_cycles,
                           dram_read_cycles, dram_write_cycles)

        results = {
            "total_cycles": total_cycles,
            "compute_cycles": compute_cycles,
            "sram_mu_read_cycles": sram_mu_read_cycles,
            "sram_mu_write_cycles": sram_mu_write_cycles,
            "dram_read_cycles": dram_read_cycles,
            "dram_write_cycles": dram_write_cycles,
            "dram_read_bytes": dram_read_bytes,
            "dram_write_bytes": dram_write_bytes,
            "sram_mu_read_bytes": sram_mu_read_bytes,
            "sram_mu_write_bytes": sram_mu_write_bytes,
            "tile_counts": {
                "total_tiles": total_tiles,
                "tiles_m": tm,
                "tiles_k": tk,
                "tiles_n": tn
            }
        }
        return results

    def print_report(self, results):
        # dump sim results
        print("total execution cycles:", results["total_cycles"])
        print("compute cycles (matrix units):", results["compute_cycles"])
        print("dram->sram transfer cycles: read =", results["dram_read_cycles"],
              "cycles, write =", results["dram_write_cycles"], "cycles")
        print("sram->matrix unit transfer cycles: read =", results["sram_mu_read_cycles"],
              "cycles, write =", results["sram_mu_write_cycles"], "cycles")
        print("\ntotal bytes transferred:")
        print("  dram -> sram (read):", results["dram_read_bytes"], "bytes")
        print("  sram -> dram (write):", results["dram_write_bytes"], "bytes")
        print("  sram -> matrix unit (read):", results["sram_mu_read_bytes"], "bytes")
        print("  matrix unit -> sram (write):", results["sram_mu_write_bytes"], "bytes")
        print("\ntile breakdown (mu-level tiling):")
        tc = results["tile_counts"]
        print("  total tiles:", tc["total_tiles"])
        print("  tiles along m:", tc["tiles_m"], " | k:", tc["tiles_k"], " | n:", tc["tiles_n"])

def parse_args():
    # expect 3 args: M, K, N
    if len(sys.argv) != 4:
        print("usage: python dnn_simulator.py <M> <K> <N>")
        sys.exit(1)
    try:
        M = int(sys.argv[1])
        K = int(sys.argv[2])
        N = int(sys.argv[3])
    except ValueError:
        print("error: M, K, and N must be integers")
        sys.exit(1)

        # dims must be multiple of 8
    if M % 8 != 0 or K % 8 != 0 or N % 8 != 0:
        print("warning: dims should be multiples of 8 for proper tiling")
    return M, K, N

def main():
    M, K, N = parse_args()
    sim = DNNSim(M, K, N)
    results = sim.run_simulation()
    sim.print_report(results)

if __name__ == "__main__":
    main()
