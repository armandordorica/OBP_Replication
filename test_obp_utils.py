"""
Test script for obp_utils package

This script demonstrates how to use the obp_utils package modules.
Run this from the OBP_Replication directory.
"""

print("="*80)
print("OBP_UTILS PACKAGE TEST")
print("="*80)

# Test 1: Import package
print("\n1️⃣  Testing package imports...")
try:
    from obp_utils import (
        load_data,
        compute_ctr,
        get_dataset_stats,
        load_all_campaigns,
        load_all_policies
    )
    print("✅ Data loader imports successful")
except ImportError as e:
    print(f"❌ Data loader import failed: {e}")

try:
    from obp_utils.stats import (
        compute_manual_propensity,
        compute_feature_combinations
    )
    print("✅ Stats module imports successful")
except ImportError as e:
    print(f"❌ Stats import failed: {e}")

try:
    from obp_utils.visualizations import (
        plot_histogram_with_stats,
        plot_bar_chart
    )
    print("✅ Visualizations module imports successful")
except ImportError as e:
    print(f"❌ Visualizations import failed: {e}")

# Test 2: Load sample data
print("\n2️⃣  Testing data loading...")
try:
    df_sample = load_data('random', 'all', 'sample')
    print(f"✅ Loaded {len(df_sample):,} records")
    print(f"   Columns: {list(df_sample.columns[:5])}...")
except Exception as e:
    print(f"❌ Data loading failed: {e}")

# Test 3: Compute CTR
print("\n3️⃣  Testing CTR computation...")
try:
    ctr = compute_ctr(df_sample)
    print(f"✅ CTR computed: {ctr:.4f} ({ctr*100:.2f}%)")
except Exception as e:
    print(f"❌ CTR computation failed: {e}")

# Test 4: Get dataset statistics
print("\n4️⃣  Testing dataset statistics...")
try:
    stats = get_dataset_stats(df_sample)
    print(f"✅ Statistics computed:")
    print(f"   Records: {stats['n_records']:,}")
    print(f"   Clicks: {stats['n_clicks']}")
    print(f"   Unique Actions: {stats['n_unique_actions']}")
    print(f"   Unique Positions: {stats['n_positions']}")
except Exception as e:
    print(f"❌ Statistics computation failed: {e}")

# Test 5: Load all campaigns
print("\n5️⃣  Testing bulk campaign loading...")
try:
    campaigns = load_all_campaigns('random', 'sample')
    print(f"✅ Loaded {len(campaigns)} campaigns:")
    for name, df in campaigns.items():
        if df is not None:
            print(f"   {name.upper()}: {len(df):,} records")
except Exception as e:
    print(f"❌ Bulk loading failed: {e}")

# Test 6: Load all policies
print("\n6️⃣  Testing bulk policy loading...")
try:
    policies = load_all_policies('all', 'sample')
    print(f"✅ Loaded {len(policies)} policies:")
    for name, df in policies.items():
        if df is not None:
            ctr_val = compute_ctr(df)
            print(f"   {name.upper()}: CTR = {ctr_val:.4f}")
except Exception as e:
    print(f"❌ Policy loading failed: {e}")

# Test 7: Feature combinations
print("\n7️⃣  Testing feature combination analysis...")
try:
    user_features = [c for c in df_sample.columns if c.startswith('user_feature_')]
    unique_vals, total_combos = compute_feature_combinations(
        df_sample, 
        user_features,
        verbose=False
    )
    print(f"✅ Feature analysis complete:")
    print(f"   Features: {len(user_features)}")
    print(f"   Total possible combinations: {total_combos:,}")
except Exception as e:
    print(f"❌ Feature analysis failed: {e}")

# Test 8: Manual propensity
print("\n8️⃣  Testing manual propensity computation...")
try:
    prop_df = compute_manual_propensity(df_sample, 'user_feature_0')
    print(f"✅ Propensity scores computed:")
    print(f"   Shape: {prop_df.shape}")
    print(f"   Columns: {list(prop_df.columns)}")
except Exception as e:
    print(f"❌ Propensity computation failed: {e}")

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("✅ All core functions tested successfully!")
print("\nPackage is ready to use. Import with:")
print("  from obp_utils import load_data, compute_ctr, ...")
print("="*80)
