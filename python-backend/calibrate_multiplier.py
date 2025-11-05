"""
Calibrate the activation_multiplier based on empirical data
"""

# Empirical data from testing:
# 4GB GPU: 4000px works, 6000px causes OOM (needs ~10.5GB total)

# Calculate base image memory
def calc_base_memory_gb(dimension):
    pixels = dimension * dimension
    bytes_needed = pixels * 3 * 4  # RGB float32
    return bytes_needed / (1024**3)

# For 4000px (SAFE)
base_4000 = calc_base_memory_gb(4000)
print(f"4000px base image memory: {base_4000:.3f}GB")

# For 6000px (OOM - needs ~10.5GB total)
base_6000 = calc_base_memory_gb(6000)
print(f"6000px base image memory: {base_6000:.3f}GB")
print(f"6000px total needed: ~10.5GB (from OOM error)")

# Calculate effective multiplier for 6000px
total_6000 = 10.5
model_overhead = 0.5
available = 4.0 - model_overhead
multiplier_6000 = (total_6000 - model_overhead) / base_6000
print(f"\nEffective multiplier for 6000px: {multiplier_6000:.1f}x")

# Now work backwards: what multiplier makes 4000px safe on 4GB?
vram_4gb = 4.0
available_4gb = vram_4gb - model_overhead
safety_margin = 0.2
usable_4gb = available_4gb * (1 - safety_margin)

# For 4000px to be safe:
# usable / multiplier = base_4000
# multiplier = usable / base_4000
required_multiplier = usable_4gb / base_4000
print(f"\nRequired multiplier for 4000px to be safe on 4GB: {required_multiplier:.1f}x")

# Average of both
avg_multiplier = (multiplier_6000 + required_multiplier) / 2
print(f"\nRecommended activation_multiplier: ~{avg_multiplier:.0f}x")

# Test with recommended multiplier
print(f"\n{'='*60}")
print("VERIFICATION with multiplier = {:.0f}".format(avg_multiplier))
print(f"{'='*60}")

for vram in [2.0, 4.0, 6.0, 8.0]:
    available = vram - model_overhead
    usable = available * (1 - safety_margin)
    image_budget = usable / avg_multiplier

    # Calculate max dimension
    import math
    pixels = (image_budget * (1024**3)) / (3 * 4)
    dimension = int(math.sqrt(pixels))
    dimension = (dimension // 1000) * 1000  # Round down

    print(f"{vram:.1f}GB VRAM -> {dimension}px")

print(f"\nExpected: 4GB -> 4000px (should match empirical safe limit)")
