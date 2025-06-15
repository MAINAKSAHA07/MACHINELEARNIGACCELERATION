import sys
import math

class MatrixUnit:
    """Represents a single matrix unit in the DNN accelerator"""
    
    def __init__(self):
        self.tile_m = 8
        self.tile_k = 8
        self.tile_n = 8
        self.cycles_per_tile = 16


class Memory:
    """Base class for memory components"""
    
    def __init__(self, size, read_bw, write_bw):
        self.size = size
        self.read_bw = read_bw
        self.write_bw = write_bw


class SRAM(Memory):
    """Represents the SRAM in the DNN accelerator"""
    
    def __init__(self):
        # 2MB SRAM with specified bandwidths
        super().__init__(2 * 1024 * 1024, 128, 64)


class DRAM(Memory):
    """Represents the DRAM in the DNN accelerator"""
    
    def __init__(self):
        # DRAM with specified bandwidths
        super().__init__(None, 4, 2)


class Workload:
    """Represents a DNN workload with dimensions M, K, N"""
    
    def __init__(self, M, K, N):
        self.M = M
        self.K = K
        self.N = N
        self.element_size = 1  # int8 = 1 byte


class DNNAccelerator:
    """Represents the complete DNN accelerator system"""
    
    def __init__(self):
        self.matrix_units = [MatrixUnit() for _ in range(8)]
        self.sram = SRAM()
        self.dram = DRAM()
        
    @property
    def num_mu(self):
        return len(self.matrix_units)
    
    @property
    def tile_m(self):
        return self.matrix_units[0].tile_m
    
    @property
    def tile_k(self):
        return self.matrix_units[0].tile_k
    
    @property
    def tile_n(self):
        return self.matrix_units[0].tile_n
    
    @property
    def cycles_per_tile(self):
        return self.matrix_units[0].cycles_per_tile


class Simulator:
    """Core simulation logic for DNN workloads on the accelerator"""
    
    def __init__(self, workload, accelerator):
        self.workload = workload
        self.accelerator = accelerator
    
    def compute_mu_tile_counts(self):
        """
        Calculate the number of matrix unit tiles needed for the workload.
        Each tile operates on 8x8x8 matrices.
        
        Returns:
            total_tiles: Total number of 8x8x8 tiles needed
            num_tiles_m, num_tiles_k, num_tiles_n: Number of tiles in each dimension
        """
        num_tiles_m = math.ceil(self.workload.M / self.accelerator.tile_m)
        num_tiles_k = math.ceil(self.workload.K / self.accelerator.tile_k)
        num_tiles_n = math.ceil(self.workload.N / self.accelerator.tile_n)
        total_tiles = num_tiles_m * num_tiles_k * num_tiles_n
        return total_tiles, num_tiles_m, num_tiles_k, num_tiles_n
    
    def calculate_optimal_sram_tile_size(self):
        """
        Calculate the optimal tile size for DRAM-SRAM transfers to maximize SRAM usage.
        
        Returns:
            Dictionary containing tile dimensions and number of tiles needed
        """
        # Start with large tiles
        Tm, Tk, Tn = self.workload.M, self.workload.K, self.workload.N
        
        # Calculate memory requirement for input, weight, and output matrices
        # Memory required = (Tm*Tk + Tk*Tn + Tm*Tn) * element_size
        while (Tm * Tk + Tk * Tn + Tm * Tn) * self.workload.element_size > self.accelerator.sram.size:
            # Reduce the largest dimension first
            if Tm >= Tk and Tm >= Tn:
                Tm = max(self.accelerator.tile_m, Tm // 2)  # Ensure it's at least the size of matrix unit tile
            elif Tk >= Tm and Tk >= Tn:
                Tk = max(self.accelerator.tile_k, Tk // 2)
            else:
                Tn = max(self.accelerator.tile_n, Tn // 2)

        # Calculate number of SRAM tiles
        num_sram_tiles_m = math.ceil(self.workload.M / Tm)
        num_sram_tiles_k = math.ceil(self.workload.K / Tk)
        num_sram_tiles_n = math.ceil(self.workload.N / Tn)
        total_sram_tiles = num_sram_tiles_m * num_sram_tiles_k * num_sram_tiles_n

        return {
            "Tm": Tm, "Tk": Tk, "Tn": Tn,
            "num_sram_tiles_m": num_sram_tiles_m,
            "num_sram_tiles_k": num_sram_tiles_k,
            "num_sram_tiles_n": num_sram_tiles_n,
            "total_sram_tiles": total_sram_tiles
        }
    
    def compute_compute_cycles(self, total_mu_tiles):
        """
        Calculate the total compute cycles needed.
        
        Args:
            total_mu_tiles: Total number of matrix unit tiles
            
        Returns:
            Total compute cycles
        """
        # Each matrix unit can process one 8x8x8 tile in 16 cycles
        # With 8 matrix units working in parallel, divide by 8 and round up
        return math.ceil(total_mu_tiles / self.accelerator.num_mu) * self.accelerator.cycles_per_tile
    
    def compute_sram_mu_transfers(self, total_mu_tiles):
        """
        Calculate the cycles and bytes for SRAM-Matrix Unit data transfers.
        
        Args:
            total_mu_tiles: Total number of matrix unit tiles
            
        Returns:
            read_cycles: Cycles for reading from SRAM to Matrix Units
            write_cycles: Cycles for writing from Matrix Units to SRAM
            total_read_bytes: Total bytes read from SRAM
            total_write_bytes: Total bytes written to SRAM
        """
        # For each tile, we need to read input and weight matrices and write output matrix
        # As specified, no data reuse at this level
        bytes_read_per_tile = (self.accelerator.tile_m * self.accelerator.tile_k + 
                              self.accelerator.tile_k * self.accelerator.tile_n) * self.workload.element_size
        bytes_write_per_tile = (self.accelerator.tile_m * self.accelerator.tile_n) * self.workload.element_size

        total_read_bytes = total_mu_tiles * bytes_read_per_tile
        total_write_bytes = total_mu_tiles * bytes_write_per_tile

        read_cycles = math.ceil(total_read_bytes / self.accelerator.sram.read_bw)
        write_cycles = math.ceil(total_write_bytes / self.accelerator.sram.write_bw)
        
        return read_cycles, write_cycles, total_read_bytes, total_write_bytes
    
    def compute_dram_sram_transfers(self, sram_tile_info):
        """
        Calculate the cycles and bytes for DRAM-SRAM data transfers.
        
        Args:
            sram_tile_info: Dictionary with SRAM tile dimensions and counts
            
        Returns:
            dram_read_cycles: Cycles for reading from DRAM to SRAM
            dram_write_cycles: Cycles for writing from SRAM to DRAM
            total_dram_read_bytes: Total bytes read from DRAM
            total_dram_write_bytes: Total bytes written to DRAM
        """
        Tm, Tk, Tn = sram_tile_info["Tm"], sram_tile_info["Tk"], sram_tile_info["Tn"]
        total_dram_read_bytes = 0
        total_dram_write_bytes = 0

        # For each SRAM tile, calculate bytes transferred
        for i in range(sram_tile_info["num_sram_tiles_m"]):
            for j in range(sram_tile_info["num_sram_tiles_k"]):
                for l in range(sram_tile_info["num_sram_tiles_n"]):
                    # Handle edge cases where tiles might be smaller
                    actual_Tm = min(Tm, self.workload.M - i * Tm)
                    actual_Tk = min(Tk, self.workload.K - j * Tk)
                    actual_Tn = min(Tn, self.workload.N - l * Tn)
                    
                    # For each SRAM tile, read input and weight matrices, write output matrix
                    total_dram_read_bytes += (actual_Tm * actual_Tk + actual_Tk * actual_Tn) * self.workload.element_size
                    total_dram_write_bytes += (actual_Tm * actual_Tn) * self.workload.element_size

        dram_read_cycles = math.ceil(total_dram_read_bytes / self.accelerator.dram.read_bw)
        dram_write_cycles = math.ceil(total_dram_write_bytes / self.accelerator.dram.write_bw)

        return dram_read_cycles, dram_write_cycles, total_dram_read_bytes, total_dram_write_bytes
    
    def run_simulation(self):
        """
        Run the complete simulation and collect results.
        
        Returns:
            Dictionary containing all simulation results
        """
        # Calculate number of matrix unit tiles
        total_mu_tiles, tm, tk, tn = self.compute_mu_tile_counts()
        
        # Calculate optimal SRAM tile size
        sram_tile_info = self.calculate_optimal_sram_tile_size()
        
        # Calculate compute cycles
        compute_cycles = self.compute_compute_cycles(total_mu_tiles)

        # Calculate data transfer cycles and bytes
        sram_mu_read_cycles, sram_mu_write_cycles, sram_mu_read_bytes, sram_mu_write_bytes = self.compute_sram_mu_transfers(total_mu_tiles)
        dram_read_cycles, dram_write_cycles, dram_read_bytes, dram_write_bytes = self.compute_dram_sram_transfers(sram_tile_info)

        # Total execution time is the maximum of all cycles
        # This is based on the assumption of complete overlap of data transfers and computation
        total_cycles = max(compute_cycles, sram_mu_read_cycles, sram_mu_write_cycles, dram_read_cycles, dram_write_cycles)

        return {
            "total_cycles": total_cycles,
            "compute_cycles": compute_cycles,
            "dram_read_cycles": dram_read_cycles,
            "dram_write_cycles": dram_write_cycles,
            "sram_mu_read_cycles": sram_mu_read_cycles,
            "sram_mu_write_cycles": sram_mu_write_cycles,
            "dram_read_bytes": dram_read_bytes,
            "dram_write_bytes": dram_write_bytes,
            "sram_mu_read_bytes": sram_mu_read_bytes,
            "sram_mu_write_bytes": sram_mu_write_bytes,
            "mu_tile_counts": {"total_tiles": total_mu_tiles, "tiles_m": tm, "tiles_k": tk, "tiles_n": tn},
            "sram_tile_info": sram_tile_info
        }


class Reporter:
    """Handles reporting of simulation results"""
    
    @staticmethod
    def print_report(workload, accelerator, results):
        """
        Print the simulation results as specified in the assignment.
        
        Args:
            workload: The workload being simulated
            accelerator: The accelerator being used
            results: Dictionary containing all simulation results
        """
        print("\n--- DNN Simulator Results ---")
        print(f"Workload Dimensions: M={workload.M}, K={workload.K}, N={workload.N}")
        
        # Print required outputs as specified in the assignment
        print(f"\n1. Total execution cycles: {results['total_cycles']}")
        print(f"2. Total cycles consumed by compute (matrix units): {results['compute_cycles']}")
        print(f"3. Total cycles consumed for data transfer from DRAM to SRAM: {max(results['dram_read_cycles'], results['dram_write_cycles'])}")
        print(f"4. Total cycles consumed for data transfer from SRAM to matrix units: {max(results['sram_mu_read_cycles'], results['sram_mu_write_cycles'])}")
        print(f"5. Total bytes transferred from DRAM to SRAM (read): {results['dram_read_bytes']} bytes")
        print(f"   Total bytes transferred from SRAM to DRAM (write): {results['dram_write_bytes']} bytes")
        print(f"6. Total bytes transferred from SRAM to matrix unit (read): {results['sram_mu_read_bytes']} bytes")
        print(f"   Total bytes transferred from matrix unit to SRAM (write): {results['sram_mu_write_bytes']} bytes")
        
        # Additional useful information
        print("\n--- Additional Information ---")
        print(f"SRAM Tile Size (DRAM to SRAM): Tm={results['sram_tile_info']['Tm']}, Tk={results['sram_tile_info']['Tk']}, Tn={results['sram_tile_info']['Tn']}")
        print(f"Number of SRAM Tiles: {results['sram_tile_info']['total_sram_tiles']}")
        print(f"Matrix Unit Tile Size (SRAM to Matrix Unit): m={accelerator.tile_m}, k={accelerator.tile_k}, n={accelerator.tile_n}")
        print(f"Number of Matrix Unit Tiles: {results['mu_tile_counts']['total_tiles']}")
        
        # Calculate peak compute capabilities
        # Each matrix unit performs 8x8x8 = 512 MACs in 16 cycles
        # Each MAC is 2 operations (1 multiply, 1 add)
        peak_ops_per_cycle = accelerator.num_mu * (2 * accelerator.tile_m * accelerator.tile_k * accelerator.tile_n) / accelerator.cycles_per_tile
        print(f"Peak compute throughput: {peak_ops_per_cycle} OPS/cycle")
        
        # Calculate achieved utilization
        total_ops = 2 * workload.M * workload.K * workload.N  # Each MAC is 2 ops
        achieved_ops_per_cycle = total_ops / results['total_cycles']
        utilization = (achieved_ops_per_cycle / peak_ops_per_cycle) * 100
        print(f"Compute unit utilization: {utilization:.2f}%")


def parse_args():
    """
    Parse command line arguments (M, K, N).
    
    Returns:
        M, K, N: Workload dimensions
    """
    if len(sys.argv) != 4:
        print("Usage: python dnnsim.py <M> <K> <N>")
        print("Example: python dnnsim.py 4096 4096 4096")
        sys.exit(1)
    try:
        M, K, N = map(int, sys.argv[1:])
        return M, K, N
    except ValueError:
        print("Error: M, K, and N must be integers")
        sys.exit(1)


def main():
    """
    Main function to run the DNN simulator.
    """
    M, K, N = parse_args()
    print(f"Running simulation for M={M}, K={K}, N={N}...")
    
    # Initialize components
    workload = Workload(M, K, N)
    accelerator = DNNAccelerator()
    
    # Create and run simulator
    simulator = Simulator(workload, accelerator)
    results = simulator.run_simulation()
    
    # Report results
    Reporter.print_report(workload, accelerator, results)


if __name__ == "__main__":
    main()