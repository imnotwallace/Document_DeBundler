============================================================================================= test session starts ==============================================================================================
platform win32 -- Python 3.12.12, pytest-7.4.4, pluggy-1.6.0 -- F:\Document-De-Bundler\python-backend\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: F:\Document-De-Bundler\python-backend
plugins: anyio-4.11.0, cov-4.1.0
collected 0 items / 1 error

==================================================================================================== ERRORS ====================================================================================================
_______________________________________________________________________________ ERROR collecting tests/test_partial_ocr_fixes.py _______________________________________________________________________________
ImportError while importing test module 'F:\Document-De-Bundler\python-backend\tests\test_partial_ocr_fixes.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\ProgramData\miniconda3\envs\document-debundler\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\test_partial_ocr_fixes.py:8: in <module>
    from services.ocr_batch_service import OCRBatchService
E   ModuleNotFoundError: No module named 'services'
=============================================================================================== warnings summary ===============================================================================================
<frozen importlib._bootstrap>:488
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================================================================================== short test summary info ============================================================================================
ERROR tests/test_partial_ocr_fixes.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
========================================================================================= 5 warnings, 1 error in 0.24s =========================================================================================
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
(.venv) (base) PS F:\Document-De-Bundler\python-backend> pytest tests/test_partial_ocr_fixes.py -v
============================================================================================= test session starts ==============================================================================================
platform win32 -- Python 3.12.12, pytest-7.4.4, pluggy-1.6.0 -- F:\Document-De-Bundler\python-backend\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: F:\Document-De-Bundler\python-backend
plugins: anyio-4.11.0, cov-4.1.0
collected 7 items

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_partial_coverage_detection PASSED                                                                                                              [ 14%]
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication PASSED                                                                                                                     [ 28%]
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation FAILED                                                                                                                   [ 42%]
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_quality_preservation FAILED                                                                                                                    [ 57%]
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_empty_coverage_metrics_fallback PASSED                                                                                                         [ 71%]
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages FAILED                                                                                                            [ 85%]
tests/test_partial_ocr_fixes.py::TestIntegration::test_end_to_end_workflow PASSED                                                                                                                         [100%]

=================================================================================================== FAILURES ===================================================================================================
________________________________________________________________________________ TestPartialOCRFixes.test_ocr_output_validation ________________________________________________________________________________

self = <test_partial_ocr_fixes.TestPartialOCRFixes object at 0x000002ADE6D3E510>
sample_partial_pdf = WindowsPath('C:/Users/samue.SAM-NITRO5/AppData/Local/Temp/pytest-of-samue/pytest-2/test_ocr_output_validation0/partial_coverage_test.pdf')
temp_dir = WindowsPath('C:/Users/samue.SAM-NITRO5/AppData/Local/Temp/pytest-of-samue/pytest-2/test_ocr_output_validation0/ocr_test_outputs')

    def test_ocr_output_validation(self, sample_partial_pdf, temp_dir):
        """
        Test that OCR output is validated and empty results are rejected.
        """
        service = OCRBatchService(use_gpu=False)

        try:
            # Process
            result = service.process_batch(
                files=[str(sample_partial_pdf)],
                output_dir=str(temp_dir)
            )

            # Check stats
            assert result['successful'], "Processing should succeed"
            file_result = result['successful'][0]

            # Should have attempted OCR
>           assert file_result['pages_ocr'] > 0, "Should have OCR'd at least one page"
E           AssertionError: Should have OCR'd at least one page
E           assert 0 > 0

tests\test_partial_ocr_fixes.py:198: AssertionError
--------------------------------------------------------------------------------------------- Captured stderr call ---------------------------------------------------------------------------------------------
Creating model: ('PP-LCNet_x1_0_doc_ori', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `C:\Users\samue.SAM-NITRO5\.paddlex\official_models\PP-LCNet_x1_0_doc_ori`.
Creating model: ('PP-LCNet_x1_0_doc_ori', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `C:\Users\samue.SAM-NITRO5\.paddlex\official_models\PP-LCNet_x1_0_doc_ori`.
---------------------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------------------
ERROR    services.ocr.engines.paddleocr_engine:paddleocr_engine.py:115 Failed to initialize PaddleOCR: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
ERROR    services.ocr.manager:manager.py:70 Failed to initialize paddleocr: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:603 Page 1: OCR validation failed - Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.paddleocr_engine:paddleocr_engine.py:115 Failed to initialize PaddleOCR: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
ERROR    services.ocr.manager:manager.py:70 Failed to initialize paddleocr: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
ERROR    services.ocr_service:ocr_service.py:53 Failed to initialize OCR service: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr_service.py", line 50, in __init__
    self.manager.initialize()
  File "F:\Document-De-Bundler\python-backend\services\ocr\manager.py", line 65, in initialize
    self.engine.initialize()
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
WARNING  services.ocr_service:ocr_service.py:99 OCR not initialized
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 PaddleOCR 400 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr_service:ocr_service.py:106 OCR error: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 Tesseract 300 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr_service:ocr_service.py:106 OCR error: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 Tesseract 400 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr_batch_service:ocr_batch_service.py:377 All OCR retry strategies failed for page 1
ERROR    services.ocr_batch_service:ocr_batch_service.py:613 Page 1: All OCR attempts failed
WARNING  services.ocr_batch_service:ocr_batch_service.py:761 Page 1: OCR failed, no text extracted
________________________________________________________________________________ TestPartialOCRFixes.test_quality_preservation _________________________________________________________________________________

self = <test_partial_ocr_fixes.TestPartialOCRFixes object at 0x000002ADE6D3E750>
sample_full_ocr_pdf = WindowsPath('C:/Users/samue.SAM-NITRO5/AppData/Local/Temp/pytest-of-samue/pytest-2/test_quality_preservation0/full_ocr_test.pdf')
temp_dir = WindowsPath('C:/Users/samue.SAM-NITRO5/AppData/Local/Temp/pytest-of-samue/pytest-2/test_quality_preservation0/ocr_test_outputs')

    def test_quality_preservation(self, sample_full_ocr_pdf, temp_dir):
        """
        Test that high-quality existing OCR is preserved, not re-processed.
        """
        # First, verify the PDF has good coverage
        with PDFProcessor(str(sample_full_ocr_pdf)) as pdf:
            has_valid, metrics = pdf.has_valid_text_layer(0, return_metrics=True)
            original_text = pdf.extract_text(0)

>           assert has_valid, "Full OCR PDF should pass validation"
E           AssertionError: Full OCR PDF should pass validation
E           assert False

tests\test_partial_ocr_fixes.py:223: AssertionError
____________________________________________________________________________ TestPartialOCRFixes.test_batch_processing_mixed_pages _____________________________________________________________________________

self = <test_partial_ocr_fixes.TestPartialOCRFixes object at 0x000002ADE6D3EA20>, tmp_path = WindowsPath('C:/Users/samue.SAM-NITRO5/AppData/Local/Temp/pytest-of-samue/pytest-2/test_batch_processing_mixed_pa0')
temp_dir = WindowsPath('C:/Users/samue.SAM-NITRO5/AppData/Local/Temp/pytest-of-samue/pytest-2/test_batch_processing_mixed_pa0/ocr_test_outputs')

    def test_batch_processing_mixed_pages(self, tmp_path, temp_dir):
        """
        Test processing PDF with mix of good and partial coverage pages.
        """
        # Create PDF with 3 pages:
        # Page 1: Full text layer (good)
        # Page 2: Partial text layer (needs OCR)
        # Page 3: No text layer (needs OCR)

        pdf_path = tmp_path / "mixed_coverage_test.pdf"
        doc = fitz.open()

        # Page 1: Full coverage
        page1 = doc.new_page(width=612, height=792)
        page1.insert_textbox(page1.rect, "Full text coverage page\n" * 30, fontsize=10)

        # Page 2: Partial coverage
        page2 = doc.new_page(width=612, height=792)
        page2.insert_textbox(fitz.Rect(50, 50, 562, 100), "Header only", fontsize=12)
        page2.draw_rect(fitz.Rect(50, 150, 562, 742), fill=(0.9, 0.9, 0.9))

        # Page 3: No coverage
        page3 = doc.new_page(width=612, height=792)
        page3.draw_rect(page3.rect, fill=(0.9, 0.9, 0.9))

        doc.save(str(pdf_path))
        doc.close()

        # Process
        service = OCRBatchService(use_gpu=False)

        try:
            result = service.process_batch(
                files=[str(pdf_path)],
                output_dir=str(temp_dir)
            )

            file_result = result['successful'][0]

            # Page 1 should use text layer
            # Pages 2-3 should use OCR
>           assert file_result['pages_text_layer'] == 1, "Should use text layer for page 1"
E           AssertionError: Should use text layer for page 1
E           assert 0 == 1

tests\test_partial_ocr_fixes.py:299: AssertionError
--------------------------------------------------------------------------------------------- Captured stderr call ---------------------------------------------------------------------------------------------
Creating model: ('PP-LCNet_x1_0_doc_ori', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `C:\Users\samue.SAM-NITRO5\.paddlex\official_models\PP-LCNet_x1_0_doc_ori`.
Creating model: ('PP-LCNet_x1_0_doc_ori', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `C:\Users\samue.SAM-NITRO5\.paddlex\official_models\PP-LCNet_x1_0_doc_ori`.
Creating model: ('PP-LCNet_x1_0_doc_ori', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `C:\Users\samue.SAM-NITRO5\.paddlex\official_models\PP-LCNet_x1_0_doc_ori`.
Creating model: ('PP-LCNet_x1_0_doc_ori', None)
Model files already exist. Using cached files. To redownload, please delete the directory manually: `C:\Users\samue.SAM-NITRO5\.paddlex\official_models\PP-LCNet_x1_0_doc_ori`.
---------------------------------------------------------------------------------------------- Captured log call -----------------------------------------------------------------------------------------------
ERROR    services.ocr.engines.paddleocr_engine:paddleocr_engine.py:115 Failed to initialize PaddleOCR: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
ERROR    services.ocr.manager:manager.py:70 Failed to initialize paddleocr: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:603 Page 1: OCR validation failed - Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.paddleocr_engine:paddleocr_engine.py:115 Failed to initialize PaddleOCR: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
ERROR    services.ocr.manager:manager.py:70 Failed to initialize paddleocr: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
ERROR    services.ocr_service:ocr_service.py:53 Failed to initialize OCR service: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr_service.py", line 50, in __init__
    self.manager.initialize()
  File "F:\Document-De-Bundler\python-backend\services\ocr\manager.py", line 65, in initialize
    self.engine.initialize()
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
WARNING  services.ocr_service:ocr_service.py:99 OCR not initialized
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 PaddleOCR 400 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr_service:ocr_service.py:106 OCR error: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 Tesseract 300 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr_service:ocr_service.py:106 OCR error: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 Tesseract 400 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr_batch_service:ocr_batch_service.py:377 All OCR retry strategies failed for page 1
ERROR    services.ocr_batch_service:ocr_batch_service.py:613 Page 1: All OCR attempts failed
WARNING  services.ocr_batch_service:ocr_batch_service.py:603 Page 2: OCR validation failed - Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.paddleocr_engine:paddleocr_engine.py:115 Failed to initialize PaddleOCR: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
ERROR    services.ocr.manager:manager.py:70 Failed to initialize paddleocr: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
ERROR    services.ocr_service:ocr_service.py:53 Failed to initialize OCR service: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr_service.py", line 50, in __init__
    self.manager.initialize()
  File "F:\Document-De-Bundler\python-backend\services\ocr\manager.py", line 65, in initialize
    self.engine.initialize()
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
WARNING  services.ocr_service:ocr_service.py:99 OCR not initialized
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 PaddleOCR 400 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr_service:ocr_service.py:106 OCR error: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 Tesseract 300 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr_service:ocr_service.py:106 OCR error: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 Tesseract 400 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr_batch_service:ocr_batch_service.py:377 All OCR retry strategies failed for page 2
ERROR    services.ocr_batch_service:ocr_batch_service.py:613 Page 2: All OCR attempts failed
WARNING  services.ocr_batch_service:ocr_batch_service.py:603 Page 3: OCR validation failed - Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.paddleocr_engine:paddleocr_engine.py:115 Failed to initialize PaddleOCR: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
ERROR    services.ocr.manager:manager.py:70 Failed to initialize paddleocr: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
ERROR    services.ocr_service:ocr_service.py:53 Failed to initialize OCR service: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr_service.py", line 50, in __init__
    self.manager.initialize()
  File "F:\Document-De-Bundler\python-backend\services\ocr\manager.py", line 65, in initialize
    self.engine.initialize()
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py", line 92, in initialize
    self.ocr = PaddleOCR(**ocr_kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\ocr.py", line 163, in __init__
    super().__init__(**base_params)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 67, in __init__
    self.paddlex_pipeline = self._create_paddlex_pipeline()
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddleocr\_pipelines\base.py", line 105, in _create_paddlex_pipeline
    return create_pipeline(config=self._merged_paddlex_config, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\ocr\pipeline.py", line 76, in __init__
    self.doc_preprocessor_pipeline = self.create_pipeline(
                                     ^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 140, in create_pipeline
    pipeline = create_pipeline(
               ^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\__init__.py", line 167, in create_pipeline
    pipeline = BasePipeline.get(pipeline_name)(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\utils\deps.py", line 206, in _wrapper
    return old_init_func(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 103, in __init__
    self._pipeline = self._create_internal_pipeline(config, self.device)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\_parallel.py", line 158, in _create_internal_pipeline
    return self._pipeline_cls(
           ^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\doc_preprocessor\pipeline.py", line 69, in __init__
    self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\pipelines\base.py", line 106, in create_model
    model = create_predictor(
            ^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\__init__.py", line 84, in create_predictor
    return BasePredictor.get(model_name)(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 49, in __init__
    self.preprocessors, self.infer, self.postprocessors = self._build()
                                                          ^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\image_classification\predictor.py", line 82, in _build
    infer = self.create_static_infer()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\base\predictor\base_predictor.py", line 301, in create_static_infer
    return PaddleInfer(
           ^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 284, in __init__
    self.predictor = self._create()
                     ^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddlex\inference\models\common\static_infer.py", line 467, in _create
    config.set_optimization_level(3)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'paddle.base.libpaddle.AnalysisConfig' object has no attribute 'set_optimization_level'. Did you mean: 'tensorrt_optimization_level'?
WARNING  services.ocr_service:ocr_service.py:99 OCR not initialized
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 PaddleOCR 400 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr_service:ocr_service.py:106 OCR error: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 Tesseract 300 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr.engines.tesseract_engine:tesseract_engine.py:156 Tesseract OCR processing failed: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
Traceback (most recent call last):
  File "F:\Document-De-Bundler\python-backend\services\ocr\engines\tesseract_engine.py", line 120, in process_image
    data = pytesseract.image_to_data(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 596, in image_to_data
    return {
           ^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 602, in <lambda>
    Output.DICT: lambda: file_to_dict(run_and_get_output(*args), '\t', -1),
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 352, in run_and_get_output
    run_tesseract(**kwargs)
  File "F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\pytesseract\pytesseract.py", line 284, in run_tesseract
    raise TesseractError(proc.returncode, get_errors(error_string))
pytesseract.pytesseract.TesseractError: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
ERROR    services.ocr_service:ocr_service.py:106 OCR error: (1, 'Error opening data file F:\\Document-De-Bundler\\python-backend\\bin\\tesseract\\tessdata/en.traineddata Please make sure the TESSDATA_PREFIX environment variable is set to your "tessdata" directory. Failed loading language \'en\' Tesseract couldn\'t load any languages! Could not initialize tesseract.')
WARNING  services.ocr_batch_service:ocr_batch_service.py:370 Tesseract 400 DPI failed validation: Insufficient text (0 chars < 50 required)
ERROR    services.ocr_batch_service:ocr_batch_service.py:377 All OCR retry strategies failed for page 3
ERROR    services.ocr_batch_service:ocr_batch_service.py:613 Page 3: All OCR attempts failed
WARNING  services.ocr_batch_service:ocr_batch_service.py:761 Page 1: OCR failed, no text extracted
WARNING  services.ocr_batch_service:ocr_batch_service.py:761 Page 2: OCR failed, no text extracted
WARNING  services.ocr_batch_service:ocr_batch_service.py:761 Page 3: OCR failed, no text extracted
=============================================================================================== warnings summary ===============================================================================================
<frozen importlib._bootstrap>:488
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:488
<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:488
  <frozen importlib._bootstrap>:488: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
  F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py:92: DeprecationWarning: The parameter `use_angle_cls` has been deprecated and will be removed in the future. Please use `use_textline_orientation` instead.
    self.ocr = PaddleOCR(**ocr_kwargs)

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
  F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py:92: DeprecationWarning: The parameter `det_model_dir` has been deprecated and will be removed in the future. Please use `text_detection_model_dir` instead.
    self.ocr = PaddleOCR(**ocr_kwargs)

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
  F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py:92: DeprecationWarning: The parameter `rec_model_dir` has been deprecated and will be removed in the future. Please use `text_recognition_model_dir` instead.
    self.ocr = PaddleOCR(**ocr_kwargs)

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages
  F:\Document-De-Bundler\python-backend\services\ocr\engines\paddleocr_engine.py:92: DeprecationWarning: The parameter `cls_model_dir` has been deprecated and will be removed in the future. Please use `textline_orientation_model_dir` instead.
    self.ocr = PaddleOCR(**ocr_kwargs)

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\google\protobuf\internal\well_known_types.py:91: DeprecationWarning: datetime.datetime.utcfromtimestamp() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.fromtimestamp(timestamp, datetime.UTC).
    _EPOCH_DATETIME_NAIVE = datetime.datetime.utcfromtimestamp(0)

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:18: DeprecationWarning: Call to deprecated create function FileDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    DESCRIPTOR = _descriptor.FileDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:36: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:43: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:50: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:57: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:64: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:29: DeprecationWarning: Call to deprecated create function Descriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _SLOT = _descriptor.Descriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:95: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:102: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:88: DeprecationWarning: Call to deprecated create function Descriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _MULTISLOTDESC = _descriptor.Descriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:133: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:140: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:147: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:154: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:161: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:168: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:175: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:182: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:189: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:196: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:203: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:210: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:217: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:224: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:231: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:238: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:245: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:252: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:259: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:266: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:273: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:280: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:126: DeprecationWarning: Call to deprecated create function Descriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _GRAPHCONFIG = _descriptor.Descriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:311: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:318: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:325: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:332: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:339: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:346: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:353: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:360: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:367: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:374: DeprecationWarning: Call to deprecated create function FieldDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.FieldDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\data_feed_pb2.py:304: DeprecationWarning: Call to deprecated create function Descriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _DATAFEEDDESC = _descriptor.Descriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:19: DeprecationWarning: Call to deprecated create function FileDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    DESCRIPTOR = _descriptor.FileDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:33: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:37: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:41: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:45: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:49: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:53: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:57: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:61: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:65: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:69: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:73: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:77: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:81: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:85: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:89: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:93: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:97: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:101: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:27: DeprecationWarning: Call to deprecated create function EnumDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _ATTRTYPE = _descriptor.EnumDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:140: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:144: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:148: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:152: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:134: DeprecationWarning: Call to deprecated create function EnumDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _SCALAR_TYPE = _descriptor.EnumDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:170: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:174: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:178: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:182: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:186: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:190: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:194: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:198: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:202: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:206: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:210: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:214: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:218: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:222: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:226: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:230: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:234: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:238: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:242: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:246: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:250: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:254: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:258: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:262: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:266: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:270: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:274: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:278: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:282: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:286: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\paddle\base\proto\framework_pb2.py:290: DeprecationWarning: Call to deprecated create function EnumValueDescriptor(). Note: Create unlinked descriptors is going to go away. Please use get/find descriptors from generated code or query the descriptor_pool.
    _descriptor.EnumValueDescriptor(

tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_no_text_duplication
  F:\Document-De-Bundler\python-backend\.venv\Lib\site-packages\astor\op_util.py:92: DeprecationWarning: ast.Num is deprecated and will be removed in Python 3.14; use ast.Constant instead
    precedence_data = dict((getattr(ast, x, None), z) for x, y, z in op_data)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================================================================================== short test summary info ============================================================================================
FAILED tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_ocr_output_validation - AssertionError: Should have OCR'd at least one page
FAILED tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_quality_preservation - AssertionError: Full OCR PDF should pass validation
FAILED tests/test_partial_ocr_fixes.py::TestPartialOCRFixes::test_batch_processing_mixed_pages - AssertionError: Should use text layer for page 1
================================================================================== 3 failed, 4 passed, 139 warnings in 15.16s ==================================================================================
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute