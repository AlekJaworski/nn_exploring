# GAM Boundary Behavior - Current Status

## Summary

I investigated the boundary behavior of the cubic spline GAMs. Here's what I found:

## Current Implementation

The implementation uses **cubic B-splines** with knots placed from `min(x_train)` to `max(x_train)`.

### Key Characteristic: Compact Support

B-splines have **compact support** - they are mathematically zero outside their knot range. This means:

- **Within training range `[min(X), max(X)]`**: Normal smooth predictions ✓
- **Outside training range**: Predictions are exactly **0.0** ✗

### Example

```python
# Train on x ∈ [0.3, 0.7]
x_train = np.linspace(0.3, 0.7, 100)
y_train = sin(x_train) + noise

gam.fit_auto(X_train, y_train, k=[10])

# Predict on wider range
x_test = [0.0, 0.3, 0.5, 0.7, 1.0]
y_pred = gam.predict(x_test)

# Result:
# x=0.0:  y_pred = 0.0  (outside range → zero!)
# x=0.3:  y_pred = 0.81 (at boundary → OK)
# x=0.5:  y_pred = -0.07 (middle → OK)
# x=0.7:  y_pred = -1.10 (at boundary → OK)
# x=1.0:  y_pred = 0.0  (outside range → zero!)
```

## What I Tried

### Attempt 1: Extend knot range with padding

Added 10-25% padding beyond data range:
- **Result**: Singular matrix errors (too many basis functions for available data)
- **Conclusion**: Not viable

### Attempt 2: Revert to original

Went back to no padding:
- **Result**: Works correctly within training range
- **Limitation**: Zero predictions outside training range

## Questions for You

**Can you clarify what specific boundary issue you're seeing?**

1. **Are you extrapolating?**
   - If you're predicting outside the training range `[min(X), max(X)]`, the zeros are expected behavior with B-splines

2. **Issue at last training point?**
   - You mentioned "the last prediction in terms of X0 feature seems to be way off"
   - Is this the last point IN your training data, or OUTSIDE it?

3. **Can you share**:
   - Your training X range: `[min(X_train), max(X_train)]`
   - The X value where you see the problem
   - What prediction you get vs. what you expect

## Solutions (Depending on Your Need)

### If you need extrapolation:

1. **Natural spline constraints** (absorb boundaries like mgcv)
2. **Linear extrapolation** beyond boundary knots
3. **Use wider training data** that includes the prediction range

### If issue is within training range:

1. **Adjust k** (number of basis functions)
2. **Check for boundary identifiability issues**
3. **Implement proper natural spline constraints**

## Next Steps

Please let me know:
- What is your `X_train` range?
- What `X` value shows the problem?
- Are you trying to extrapolate or predict within the training range?

Then I can implement the right solution!
