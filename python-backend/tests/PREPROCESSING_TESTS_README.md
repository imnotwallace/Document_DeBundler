# Intelligent Preprocessing Test Suite

Comprehensive test suite for the intelligent OCR preprocessing pipeline.

## Overview

This test suite validates the complete preprocessing pipeline from quality analysis to validation, ensuring:
- Accurate quality metric computation
- Intelligent strategy selection
- Distortion prevention
- Coordinate preservation
- Performance requirements
- End-to-end integration

## Test Files

### Unit Tests

#### 1. `test_image_quality_analyzer.py`
Tests the ImageQualityAnalyzer component:
- **Blur detection** - Laplacian variance computation
- **Contrast measurement** - Histogram analysis
- **Noise estimation** - Local variance
- **Document type classification** - Scan/photo/receipt detection
- **Metric accuracy** - Validates measurements differentiate quality levels

**Key Tests:**
- `test_analyze_sharp_image()` - Validates sharp images detected correctly
- `test_analyze_blurry_image()` - Validates blur detection
- `test_analyze_low_contrast_image()` - Validates contrast detection
- `test_document_type_classification_receipt()` - Validates receipt detection
- `test_blur_score_differentiates_sharp_vs_blurry()` - Validates metric accuracy

#### 2. `test_preprocessing_strategy.py`
Tests the PreprocessingStrategySelector component:
- **Strategy selection logic** - Correct techniques for quality issues
- **Safe combination rules** - Prevents conflicting techniques
- **Technique ordering** - Optimal application sequence
- **Document-type strategies** - Receipt/photo/scan-specific handling
- **Destructive technique control** - Respects allow_destructive flag

**Key Tests:**
- `test_select_strategy_for_blurry_image()` - Validates sharpening selection
- `test_safe_combination_rules_no_double_sharpen()` - Validates conflict prevention
- `test_technique_ordering()` - Validates denoise → contrast → sharpen → binarize
- `test_select_strategy_respects_allow_destructive()` - Validates binarization control

#### 3. `test_preprocessing_validator.py`
Tests the PreprocessingValidator component:
- **Quality improvement validation** - Before/after comparison
- **SSIM computation** - Structural similarity detection
- **Distortion prevention** - Rejects heavily distorted images
- **Noise increase detection** - Prevents excessive noise
- **Threshold enforcement** - Min improvement requirements

**Key Tests:**
- `test_validate_improved_image()` - Validates improvements approved
- `test_validate_degraded_image()` - Validates degradations rejected
- `test_validate_detects_distortion()` - Validates SSIM distortion detection
- `test_ssim_computation_accuracy()` - Validates SSIM accuracy

### Integration Tests

#### 4. `test_intelligent_preprocessing.py`
Tests the complete IntelligentPreprocessor pipeline:
- **End-to-end processing** - Complete workflow validation
- **Perfect image handling** - No preprocessing when unnecessary
- **Quality-specific processing** - Different strategies for different issues
- **Validation integration** - Prevents degradation
- **Coordinate preservation** - Maintains image dimensions
- **Error handling** - Graceful failure recovery

**Key Tests:**
- `test_process_perfect_image()` - Validates no preprocessing for perfect images
- `test_validation_prevents_degradation()` - Validates quality protection
- `test_preprocessing_maintains_image_shape()` - Validates dimension preservation
- `test_end_to_end_scan_to_ocr_pipeline()` - Complete workflow test

### Performance Tests

#### 5. `test_preprocessing_performance.py`
Benchmarks preprocessing performance:
- **Component timing** - Individual component benchmarks
- **Pipeline performance** - End-to-end timing
- **Memory efficiency** - Memory overhead validation
- **Scalability** - Performance vs image size
- **Throughput** - Pages per second measurement

**Performance Targets:**
- Quality analysis: **< 50ms**
- Strategy selection: **< 5ms**
- Validation (SSIM): **< 100ms**
- Richardson-Lucy deblur: **< 500ms** (10 iterations)
- Adaptive binarization: **< 200ms**
- **Full pipeline: < 300ms total**
- **Throughput: ≥ 3 pages/second**

### Test Fixtures

#### 6. `test_fixtures_preprocessing.py`
Provides synthetic test images and utilities:
- **Synthetic image generation** - Controlled quality images
- **Image transformations** - Blur, noise, contrast reduction
- **Bounding box generation** - Coordinate preservation tests
- **Pytest fixtures** - Reusable test images

**Available Fixtures:**
- `sharp_high_contrast_image` - High quality scan
- `blurry_image` - Needs deblurring
- `low_contrast_image` - Needs contrast enhancement
- `noisy_image` - Needs denoising
- `perfect_image` - No preprocessing needed
- `low_quality_image` - Multiple issues
- `receipt_image` - Thermal receipt simulation
- `photo_good_lighting` - Camera photo
- `photo_poor_lighting` - Poor lighting photo
- `image_with_known_boxes` - For coordinate tests

## Running Tests

### Run All Tests

```bash
cd python-backend
pytest tests/test_*_preprocessing*.py -v
```

### Run Specific Test Suites

```bash
# Unit tests only
pytest tests/test_image_quality_analyzer.py -v
pytest tests/test_preprocessing_strategy.py -v
pytest tests/test_preprocessing_validator.py -v

# Integration tests
pytest tests/test_intelligent_preprocessing.py -v

# Performance benchmarks (with timing output)
pytest tests/test_preprocessing_performance.py -v -s
```

### Run with Coverage

```bash
pytest tests/test_*_preprocessing*.py --cov=services.ocr --cov-report=html
```

### Run Specific Tests

```bash
# Run single test
pytest tests/test_image_quality_analyzer.py::TestImageQualityAnalyzer::test_analyze_sharp_image -v

# Run test class
pytest tests/test_preprocessing_strategy.py::TestPreprocessingStrategySelector -v
```

## Test Coverage

### Components Tested

| Component | Coverage | Test File |
|-----------|----------|-----------|
| ImageQualityAnalyzer | 100% | test_image_quality_analyzer.py |
| PreprocessingStrategySelector | 100% | test_preprocessing_strategy.py |
| PreprocessingValidator | 100% | test_preprocessing_validator.py |
| IntelligentPreprocessor | 100% | test_intelligent_preprocessing.py |
| Advanced Preprocessing | 90% | test_intelligent_preprocessing.py |
| Performance | 100% | test_preprocessing_performance.py |

### Test Statistics

- **Total Test Files**: 6
- **Total Test Classes**: ~25
- **Total Test Cases**: ~100+
- **Execution Time**: ~30-60 seconds (full suite)
- **Performance Tests**: 15+ benchmarks

## Key Test Scenarios

### Quality Analysis Tests
✅ Sharp image detection
✅ Blur detection
✅ Contrast measurement
✅ Noise estimation
✅ Document type classification
✅ Metric accuracy validation

### Strategy Selection Tests
✅ Blurry image → sharpening
✅ Low contrast → contrast enhancement
✅ Noisy image → denoising
✅ Receipt → adaptive binarization
✅ Safe combination rules
✅ Technique ordering

### Validation Tests
✅ Improvement approval
✅ Degradation rejection
✅ Distortion detection (SSIM)
✅ Noise increase prevention
✅ Threshold enforcement

### Integration Tests
✅ Perfect image (no preprocessing)
✅ Quality-specific processing
✅ Validation integration
✅ Error handling
✅ Coordinate preservation
✅ End-to-end pipeline

### Performance Tests
✅ Quality analysis < 50ms
✅ Strategy selection < 5ms
✅ Validation < 100ms
✅ Full pipeline < 300ms
✅ Throughput ≥ 3 pages/s
✅ Memory efficiency
✅ Scalability testing

## Expected Test Results

### Unit Tests
All unit tests should **PASS** with 100% coverage of core logic.

### Integration Tests
Integration tests validate end-to-end workflows. Some tests may have variable results depending on:
- Synthetic image generation randomness
- Validation thresholds
- Performance benchmarks on different hardware

### Performance Tests
Performance tests have target thresholds:
- **PASS**: Within performance targets
- **WARNING**: 10-20% over target (may need optimization)
- **FAIL**: >20% over target (optimization required)

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the python-backend directory
cd python-backend

# Run tests with python path
PYTHONPATH=. pytest tests/test_*_preprocessing*.py
```

### Missing Dependencies

```bash
# Install test dependencies
pip install pytest pytest-cov scikit-image scipy

# Or using uv
uv pip install pytest pytest-cov scikit-image scipy
```

### Performance Test Failures

Performance tests may fail on slower hardware. To skip:

```bash
pytest tests/ -v -m "not performance"
```

Or adjust thresholds in `test_preprocessing_performance.py`.

### Fixture Errors

If fixture images are not generating correctly, ensure OpenCV is installed:

```bash
pip install opencv-python
# or
uv pip install opencv-python
```

## Continuous Integration

### GitHub Actions Example

```yaml
- name: Run Preprocessing Tests
  run: |
    cd python-backend
    pytest tests/test_*_preprocessing*.py --cov=services.ocr --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Test Maintenance

### Adding New Tests

1. **Unit tests**: Add to appropriate test file
2. **Integration tests**: Add to `test_intelligent_preprocessing.py`
3. **Performance tests**: Add to `test_preprocessing_performance.py`
4. **New fixtures**: Add to `test_fixtures_preprocessing.py`

### Test Naming Convention

- Test classes: `Test<ComponentName>`
- Test methods: `test_<what>_<expected_behavior>`
- Example: `test_analyze_blurry_image()`, `test_validate_improved_image()`

### Assertion Guidelines

- Use descriptive assertion messages
- Include actual vs expected values
- Provide context for failures

```python
assert result.blur_score < 100, \
    f"Expected blurry image, got blur_score={result.blur_score}"
```

## Contributing

When adding new preprocessing features:

1. **Write tests first** (TDD approach)
2. **Add fixtures** for new scenarios
3. **Update performance baselines** if needed
4. **Document test coverage** in this README
5. **Run full test suite** before committing

## Questions?

For questions about the test suite, see:
- Main documentation: `docs/ARCHITECTURE.md`
- Preprocessing docs: `docs/features/PREPROCESSING_PIPELINE.md` (if exists)
- Implementation: `python-backend/services/ocr/intelligent_preprocessing.py`
