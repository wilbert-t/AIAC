#!/usr/bin/env python3
"""
Fix Dataset Overlap: Remove Elite Structures from Other Datasets
================================================================

Problem: Elite structures appear in all three datasets, causing memorization
Solution: Remove elite structures from balanced and high_quality datasets
"""

import pandas as pd
from pathlib import Path

def fix_dataset_overlap():
    """Remove elite structures from balanced and high_quality datasets"""
    
    print("🔧 FIXING DATASET OVERLAP")
    print("=" * 50)
    
    # Load all datasets
    print("📁 Loading datasets...")
    elite_df = pd.read_csv('improved_dataset_elite.csv')
    high_quality_df = pd.read_csv('improved_dataset_high_quality.csv')
    balanced_df = pd.read_csv('improved_dataset_balanced.csv')
    
    print(f"   Elite: {len(elite_df)} structures")
    print(f"   High Quality: {len(high_quality_df)} structures")
    print(f"   Balanced: {len(balanced_df)} structures")
    
    # Get elite structure IDs
    elite_structure_ids = set(elite_df['structure_id'].tolist())
    print(f"\n🎯 Elite structure IDs to remove: {len(elite_structure_ids)}")
    
    # Check overlaps before fixing
    hq_overlap = set(high_quality_df['structure_id']).intersection(elite_structure_ids)
    balanced_overlap = set(balanced_df['structure_id']).intersection(elite_structure_ids)
    
    print(f"\n⚠️  Current overlaps:")
    print(f"   High Quality ∩ Elite: {len(hq_overlap)} structures")
    print(f"   Balanced ∩ Elite: {len(balanced_overlap)} structures")
    
    # Remove elite structures from other datasets
    print(f"\n🧹 Removing overlaps...")
    
    # Remove from high quality
    high_quality_clean = high_quality_df[~high_quality_df['structure_id'].isin(elite_structure_ids)]
    removed_from_hq = len(high_quality_df) - len(high_quality_clean)
    
    # Remove from balanced
    balanced_clean = balanced_df[~balanced_df['structure_id'].isin(elite_structure_ids)]
    removed_from_balanced = len(balanced_df) - len(balanced_clean)
    
    print(f"   Removed {removed_from_hq} structures from high_quality")
    print(f"   Removed {removed_from_balanced} structures from balanced")
    
    # Create backups
    print(f"\n💾 Creating backups...")
    high_quality_df.to_csv('improved_dataset_high_quality_backup.csv', index=False)
    balanced_df.to_csv('improved_dataset_balanced_backup.csv', index=False)
    
    # Save cleaned datasets
    print(f"\n💾 Saving cleaned datasets...")
    high_quality_clean.to_csv('improved_dataset_high_quality.csv', index=False)
    balanced_clean.to_csv('improved_dataset_balanced.csv', index=False)
    
    # Final verification
    print(f"\n✅ Final dataset sizes:")
    print(f"   Elite: {len(elite_df)} structures (unchanged)")
    print(f"   High Quality: {len(high_quality_clean)} structures (was {len(high_quality_df)})")
    print(f"   Balanced: {len(balanced_clean)} structures (was {len(balanced_df)})")
    
    # Verify no more overlaps
    hq_overlap_after = set(high_quality_clean['structure_id']).intersection(elite_structure_ids)
    balanced_overlap_after = set(balanced_clean['structure_id']).intersection(elite_structure_ids)
    
    print(f"\n🎯 Overlap verification:")
    print(f"   High Quality ∩ Elite: {len(hq_overlap_after)} structures (should be 0)")
    print(f"   Balanced ∩ Elite: {len(balanced_overlap_after)} structures (should be 0)")
    
    if len(hq_overlap_after) == 0 and len(balanced_overlap_after) == 0:
        print(f"\n🎉 SUCCESS: All overlaps removed!")
        print(f"📋 Now you can safely use:")
        print(f"   - Elite dataset for testing (no memorization)")
        print(f"   - High Quality + Balanced for training (no overlap with test)")
    else:
        print(f"\n❌ ERROR: Some overlaps still remain!")
    
    # Show which elite structures were removed
    print(f"\n📋 Elite structures that were removed from other datasets:")
    for i, struct_id in enumerate(sorted(elite_structure_ids)[:10]):  # Show first 10
        if struct_id in hq_overlap or struct_id in balanced_overlap:
            print(f"   {struct_id}")
    if len(elite_structure_ids) > 10:
        print(f"   ... and {len(elite_structure_ids) - 10} more")

def verify_datasets():
    """Verify the datasets are properly separated"""
    print(f"\n🔍 DATASET VERIFICATION")
    print("=" * 30)
    
    try:
        elite_df = pd.read_csv('improved_dataset_elite.csv')
        high_quality_df = pd.read_csv('improved_dataset_high_quality.csv')
        balanced_df = pd.read_csv('improved_dataset_balanced.csv')
        
        elite_ids = set(elite_df['structure_id'])
        hq_ids = set(high_quality_df['structure_id'])
        balanced_ids = set(balanced_df['structure_id'])
        
        print(f"Dataset sizes:")
        print(f"   Elite: {len(elite_ids)}")
        print(f"   High Quality: {len(hq_ids)}")
        print(f"   Balanced: {len(balanced_ids)}")
        
        # Check all overlaps
        elite_hq_overlap = elite_ids.intersection(hq_ids)
        elite_balanced_overlap = elite_ids.intersection(balanced_ids)
        hq_balanced_overlap = hq_ids.intersection(balanced_ids)
        
        print(f"\nOverlaps:")
        print(f"   Elite ∩ High Quality: {len(elite_hq_overlap)}")
        print(f"   Elite ∩ Balanced: {len(elite_balanced_overlap)}")
        print(f"   High Quality ∩ Balanced: {len(hq_balanced_overlap)}")
        
        # Check if Structure 350 is in elite
        if 'structure_350' in elite_ids:
            print(f"\n✅ Structure 350 is in Elite dataset (will be tested)")
        else:
            print(f"\n❌ Structure 350 is NOT in Elite dataset!")
        
        # Show energy ranges
        print(f"\nEnergy ranges:")
        print(f"   Elite: {elite_df['energy'].min():.2f} to {elite_df['energy'].max():.2f}")
        print(f"   High Quality: {high_quality_df['energy'].min():.2f} to {high_quality_df['energy'].max():.2f}")
        print(f"   Balanced: {balanced_df['energy'].min():.2f} to {balanced_df['energy'].max():.2f}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # First verify current state
    print("🔍 CHECKING CURRENT DATASET STATE")
    print("=" * 40)
    verify_datasets()
    
    print(f"\n" + "="*60)
    
    # Fix overlaps
    fix_dataset_overlap()
    
    print(f"\n" + "="*60)
    
    # Verify after fix
    verify_datasets()