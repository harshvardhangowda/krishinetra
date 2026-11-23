# KrishiNetra Dataset Audit Report

**Audit Status:** PASSED

**Dataset Location:** `C:\Users\harshvardhan\OneDrive\Documents\krishnetra Aavishkar\datasets\ip102_subset`


## Executive Summary

| Metric | Value | Status |
| :--- | :--- | :--- |
| **Total Images** | 706 | OK |
| **Total Annotations** | 706 | OK |
| **Total Bounding Boxes** | 744 | OK |
| **Synthetic Placeholder Images** | 0 | CLEAN |
| **Corrupted Images** | 0 | CLEAN |
| **Missing Label Files** | 0 | CLEAN |
| **Orphan Label Files** | 0 | CLEAN |
| **Duplicate Images** | 0 | CLEAN |
| **Data Leakage (Train/Val/Test Overlap)** | 0 | CLEAN |

## Target Pest Class Breakdown

| Class ID | Pest Name | Train Boxes | Val Boxes | Test Boxes | Total Boxes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | **Army Worm** | 150 | 44 | 22 | 216 |
| `1` | **Blister Beetle** | 187 | 53 | 28 | 268 |
| `2` | **Corn Borer** | 181 | 51 | 28 | 260 |


## Split Breakdown & Integrity

| Split | Images | Labels | Valid Pairs | Corrupt | Synthetic | Label Errors |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train` | 494 | 494 | 494 | 0 | 0 | 0 |
| `val` | 141 | 141 | 141 | 0 | 0 | 0 |
| `test` | 71 | 71 | 71 | 0 | 0 | 0 |

