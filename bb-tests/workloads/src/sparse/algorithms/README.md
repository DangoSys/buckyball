# Four Tiling Strategy Algorithms for Sparse Matrix Multiplication Accelerator

This document provides a detailed introduction to four sparse matrix multiplication tiling optimization algorithms implemented in the HYTE system: Tailors, DRT, Harp, and HYTE. These algorithms employ different tiling optimization strategies for executing sparse matrix multiplication C = A × B on hardware accelerators.

## Tailors Algorithm

The Tailors algorithm is a cache-constraint-based static tiling strategy that focuses on K-dimension tiling optimization. The core idea of this algorithm is to ensure that data for each tile can fully fit into available cache space, avoiding performance degradation caused by cache overflow.

The algorithm first calculates a base tile boundary pbound, which is determined through a sliding window approach to ensure that the total data size within the window does not exceed cache capacity Bbound. Subsequently, the algorithm attempts to expand tile size using different multipliers, gradually increasing from 1.0x to 1025x, checking the tile overflow rate each time. When the number of overflowed tiles exceeds 10% of the total number of tiles, the algorithm stops searching and selects the current tile size.

The Tailors algorithm uses IJK iteration order and allocates cache resources in a 45%, 50%, 5% ratio across three different storage hierarchies. This allocation strategy optimizes the access patterns for matrices A and B, reducing memory bandwidth requirements. The algorithm's advantages lie in its simplicity and cache-friendly characteristics, but its disadvantage is that it only performs tiling in the K dimension, which may not fully utilize optimization potential in other dimensions.

## DRT Algorithm

The DRT (Dynamic Restructuring Tiling) algorithm employs a more balanced multi-dimensional tiling strategy. Unlike Tailors, the DRT algorithm performs tiling in both J and K dimensions, using JKI iteration order to optimize data access patterns.

The algorithm is based on the tile size calculated by the Tailors algorithm, determining J and K dimension tile sizes by taking square roots: jjj = J/√tilesize, kkk = K/√tilesize. This approach aims to create more balanced tile shapes, avoiding performance issues caused by one dimension being too large or too small. The DRT algorithm uses a two-stage cache resource configuration during computation: first using a 5%, 45%, 50% allocation ratio for initial calculations, then switching to a 45%, 40%, 5% configuration for actual execution.

The main advantage of the DRT algorithm is its multi-dimensional optimization capability, which can better balance access patterns across different dimensions. By simultaneously optimizing J and K dimensions, the algorithm can reduce data reuse distance and improve cache hit rates. However, its fixed square root tiling strategy may not be suitable for all types of sparse matrix patterns.

## Harp Algorithm

The Harp algorithm employs a unique single-dimension deep tiling strategy, focusing on I-dimension tiling while keeping J and K dimensions untiled. The design philosophy of this algorithm is to reduce intermediate result storage requirements by decreasing I-dimension tile size while maintaining full access to matrix B.

The algorithm sets I-dimension tile size as iii = I/tilesize, while jjj and kkk remain as full J and K sizes respectively. This strategy is particularly suitable for certain specific sparse patterns where row-direction tiling can significantly reduce memory requirements for partial accumulation. The Harp algorithm uses JKI iteration order and employs an extreme cache allocation strategy: 4.5%, 91%, 4.5%, allocating the vast majority of cache resources to intermediate layer storage.

The advantage of the Harp algorithm lies in its highly optimized capability for specific workloads, especially when matrices have specific sparse patterns. By avoiding tiling in J and K dimensions, the algorithm reduces tile management overhead. However, this strategy's applicability is relatively limited and may not be suitable for all types of sparse matrices.

## HYTE Algorithm

The HYTE (Hybrid Static-Dynamic Tiling) algorithm is the core innovation of this system, combining the advantages of static and dynamic optimization. The algorithm first determines an initial optimal tiling strategy through static analysis, then dynamically adjusts tile sizes at runtime based on actual data access patterns.

In the static optimization phase, the HYTE algorithm evaluates a large number of possible tiling configurations using a cost model, with the search space including different tile size combinations across I, J, and K dimensions. The algorithm obtains matrix characteristic parameters through sampling techniques, then uses these parameters to guide the search process. The search strategy employs a hierarchical approach: first changing only the J dimension, then only the K dimension, and finally changing both J and K dimensions simultaneously.

In the dynamic optimization phase, the HYTE algorithm monitors non-zero element distribution within each tile during execution, dynamically adjusting tile sizes through the update_T() function. The system maintains performance statistics for multiple tiling configurations, including cache utilization, bandwidth utilization, and other key metrics. When detecting that the current tiling configuration is not optimal, the algorithm selects a better-performing configuration and immediately switches to it.

The main advantage of the HYTE algorithm is its strong adaptability, capable of handling various different sparse patterns and hardware configurations. Static optimization ensures good initial performance, while the dynamic adjustment mechanism can respond to runtime changes. The algorithm uses a 5%, 50%, 45% cache allocation strategy and supports multiple iteration orders. Although the HYTE algorithm has high complexity, its excellent performance and broad applicability make it an ideal choice for handling complex sparse matrix multiplication problems.

## Algorithm Comparison and Selection

These four algorithms each have distinct characteristics: Tailors is simple and efficient but has limited optimization dimensions; DRT provides balanced multi-dimensional optimization; Harp performs excellently in specific scenarios but has a narrow scope of application; HYTE achieves the best overall performance through a hybrid optimization strategy. In practical applications, the most suitable algorithm can be selected based on specific hardware configurations, matrix characteristics, and performance requirements.
