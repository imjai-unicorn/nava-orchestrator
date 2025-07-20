# backend/services/03-external-ai/gemini-client/tests/test_multimodal_handler.py
"""
Test suite for multimodal handler
"""

import pytest
import base64
import io
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import json

# Import the models we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from multimodal_handler import (
    MediaType, ImageFormat, AudioFormat, VideoFormat, ProcessingQuality, AnalysisType,
    MediaFile, ProcessingResult, GeminiMultimodalHandler
)

class TestMediaFile:
    """Test MediaFile dataclass"""
    
    def test_media_file_creation(self):
        """Test creating a MediaFile instance"""
        content = b"test content"
        media_file = MediaFile(
            content=content,
            mime_type="text/plain",
            filename="test.txt"
        )
        
        assert media_file.content == content
        assert media_file.mime_type == "text/plain"
        assert media_file.filename == "test.txt"
        assert media_file.size == len(content)
        assert media_file.checksum is not None
    
    def test_media_file_auto_size_calculation(self):
        """Test automatic size calculation"""
        content = b"hello world"
        media_file = MediaFile(content=content, mime_type="text/plain")
        
        assert media_file.size == len(content)
    
    def test_media_file_checksum_generation(self):
        """Test checksum generation"""
        content = b"test content"
        media_file1 = MediaFile(content=content, mime_type="text/plain")
        media_file2 = MediaFile(content=content, mime_type="text/plain")
        
        # Same content should produce same checksum
        assert media_file1.checksum == media_file2.checksum
        
        # Different content should produce different checksum
        media_file3 = MediaFile(content=b"different content", mime_type="text/plain")
        assert media_file1.checksum != media_file3.checksum

class TestProcessingResult:
    """Test ProcessingResult dataclass"""
    
    def test_processing_result_creation(self):
        """Test creating a ProcessingResult instance"""
        result = ProcessingResult(
            success=True,
            content_description="Test description",
            extracted_data={"key": "value"},
            technical_details={"format": "test"},
            processing_time=1.5,
            confidence_score=0.85,
            warnings=["warning1"],
            errors=[]
        )
        
        assert result.success
        assert result.content_description == "Test description"
        assert result.extracted_data["key"] == "value"
        assert result.processing_time == 1.5
        assert result.confidence_score == 0.85

class TestGeminiMultimodalHandler:
    """Test GeminiMultimodalHandler class"""
    
    @pytest.fixture
    def handler(self):
        """Fixture providing a handler instance"""
        return GeminiMultimodalHandler()
    
    @pytest.fixture
    def sample_text_file(self):
        """Fixture providing a sample text file"""
        content = b"This is a test document content."
        return MediaFile(
            content=content,
            mime_type="text/plain",
            filename="test.txt"
        )
    
    @pytest.fixture
    def sample_image_file(self):
        """Fixture providing a sample image file"""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return MediaFile(
            content=img_bytes.getvalue(),
            mime_type="image/jpeg",
            filename="test.jpg"
        )
    
    def test_handler_initialization(self, handler):
        """Test handler initialization"""
        assert handler.max_file_size > 0
        assert len(handler.supported_formats) > 0
        assert isinstance(handler.processing_cache, dict)
    
    def test_supported_formats_loading(self, handler):
        """Test supported formats are loaded"""
        formats = handler.supported_formats
        
        # Check for common formats
        assert "image/jpeg" in formats
        assert "image/png" in formats
        assert "audio/mpeg" in formats
        assert "video/mp4" in formats
        assert "application/pdf" in formats
        assert "text/plain" in formats
    
    def test_is_image_detection(self, handler, sample_image_file):
        """Test image file detection"""
        assert handler._is_image(sample_image_file)
        
        text_file = MediaFile(content=b"text", mime_type="text/plain")
        assert not handler._is_image(text_file)
    
    def test_is_document_detection(self, handler, sample_text_file):
        """Test document file detection"""
        assert handler._is_document(sample_text_file)
        
        pdf_file = MediaFile(content=b"pdf", mime_type="application/pdf")
        assert handler._is_document(pdf_file)
        
        image_file = MediaFile(content=b"image", mime_type="image/jpeg")
        assert not handler._is_document(image_file)
    
    def test_is_supported_format(self, handler):
        """Test supported format checking"""
        assert handler._is_supported_format("image/jpeg")
        assert handler._is_supported_format("text/plain")
        assert not handler._is_supported_format("application/unknown")
    
    def test_validate_inputs_success(self, handler, sample_text_file):
        """Test successful input validation"""
        prompt = "Analyze this document"
        media_files = [sample_text_file]
        
        is_valid, errors = handler._validate_inputs(prompt, media_files)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_inputs_empty_prompt(self, handler, sample_text_file):
        """Test validation with empty prompt"""
        prompt = ""
        media_files = [sample_text_file]
        
        is_valid, errors = handler._validate_inputs(prompt, media_files)
        
        assert not is_valid
        assert "Text prompt cannot be empty" in errors
    
    def test_validate_inputs_no_media(self, handler):
        """Test validation with no media files"""
        prompt = "Test prompt"
        media_files = []
        
        is_valid, errors = handler._validate_inputs(prompt, media_files)
        
        assert not is_valid
        assert "At least one media file is required" in errors
    
    def test_validate_inputs_large_file(self, handler):
        """Test validation with oversized file"""
        prompt = "Test prompt"
        large_content = b"x" * (handler.max_file_size + 1)
        large_file = MediaFile(content=large_content, mime_type="text/plain")
        
        is_valid, errors = handler._validate_inputs(prompt, [large_file])
        
        assert not is_valid
        assert "exceeds maximum size" in " ".join(errors)
    
    def test_validate_inputs_unsupported_format(self, handler):
        """Test validation with unsupported format"""
        prompt = "Test prompt"
        unsupported_file = MediaFile(
            content=b"test", 
            mime_type="application/unsupported"
        )
        
        is_valid, errors = handler._validate_inputs(prompt, [unsupported_file])
        
        assert not is_valid
        assert "unsupported format" in " ".join(errors)
    
    @patch('multimodal_handler.Image')
    def test_prepare_image(self, mock_image, handler):
        """Test image preparation"""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.width = 200
        mock_img.height = 200
        mock_img.save = Mock()
        mock_image.open.return_value = mock_img
        
        sample_file = MediaFile(content=b"fake_image", mime_type="image/jpeg")
        
        result = handler._prepare_image(sample_file, ProcessingQuality.BALANCED)
        
        assert "processed_content" in result
        assert "processed_format" in result
        assert result["processed_format"] == "jpeg"
    
    def test_prepare_audio(self, handler):
        """Test audio preparation"""
        audio_file = MediaFile(content=b"fake_audio", mime_type="audio/mpeg")
        
        result = handler._prepare_audio(audio_file, ProcessingQuality.BALANCED)
        
        assert "processed_content" in result
        assert "processed_format" in result
        assert "duration_estimate" in result
        assert result["processed_format"] == "mpeg"
    
    def test_prepare_video(self, handler):
        """Test video preparation"""
        video_file = MediaFile(content=b"fake_video", mime_type="video/mp4")
        
        result = handler._prepare_video(video_file, ProcessingQuality.BALANCED)
        
        assert "processed_content" in result
        assert "processed_format" in result
        assert result["processed_format"] == "mp4"
        assert result["key_frames_extracted"]
    
    def test_prepare_document_success(self, handler, sample_text_file):
        """Test document preparation success"""
        result = handler._prepare_document(sample_text_file, ProcessingQuality.BALANCED)
        
        assert "processed_content" in result
        assert "processed_format" in result
        # Should extract text successfully
        assert result["processed_format"] == "text"
    
    def test_prepare_document_failure(self, handler):
        """Test document preparation failure fallback"""
        # Binary file that can't be processed as text
        binary_file = MediaFile(content=b"\x00\x01\x02", mime_type="application/octet-stream")
        
        result = handler._prepare_document(binary_file, ProcessingQuality.BALANCED)
        
        # Should fallback to base64
        assert "processed_content" in result
        assert result["extraction_failed"]
    
    def test_build_enhanced_prompt(self, handler, sample_text_file):
        """Test enhanced prompt building"""
        original_prompt = "Analyze this content"
        processed_media = [{"filename": "test.txt", "mime_type": "text/plain"}]
        
        enhanced_prompt = handler._build_enhanced_prompt(
            original_prompt, 
            processed_media, 
            AnalysisType.DETAILED,
            "Custom instructions"
        )
        
        assert original_prompt in enhanced_prompt
        assert "Analysis Type: detailed" in enhanced_prompt
        assert "Custom instructions" in enhanced_prompt
        assert "test.txt" in enhanced_prompt
    
    def test_process_with_gemini_detailed(self, handler):
        """Test Gemini processing simulation for detailed analysis"""
        prompt = "Test prompt"
        processed_media = [{"mime_type": "text/plain"}]
        
        result = handler._process_with_gemini(prompt, processed_media, AnalysisType.DETAILED)
        
        assert "description" in result
        assert "extracted_data" in result
        assert "confidence" in result
        assert result["confidence"] > 0
    
    def test_process_with_gemini_technical(self, handler):
        """Test Gemini processing simulation for technical analysis"""
        prompt = "Test prompt"
        processed_media = [{"mime_type": "image/jpeg"}]
        
        result = handler._process_with_gemini(prompt, processed_media, AnalysisType.TECHNICAL)
        
        assert "technical_details" in result
        assert "processing_metadata" in result
    
    def test_process_with_gemini_safety(self, handler):
        """Test Gemini processing simulation for safety analysis"""
        prompt = "Test prompt"
        processed_media = [{"mime_type": "text/plain"}]
        
        result = handler._process_with_gemini(prompt, processed_media, AnalysisType.SAFETY)
        
        assert "extracted_data" in result
        assert "safety_score" in result["extracted_data"]
        assert "moderation_status" in result["extracted_data"]
    
    def test_post_process_results(self, handler):
        """Test result post-processing"""
        result = {
            "description": "Test description",
            "confidence": 0.8
        }
        processed_media = [{"filename": "test.txt", "mime_type": "text/plain", "processed_content": "text"}]
        
        enhanced = handler._post_process_results(result, processed_media, AnalysisType.DETAILED)
        
        assert "media_insights" in enhanced
        assert len(enhanced["media_insights"]) == 1
        assert enhanced["media_insights"][0]["filename"] == "test.txt"
    
    def test_calculate_confidence_score(self, handler):
        """Test confidence score calculation"""
        processed_media = [
            {"processed_content": "content1"},
            {"processed_content": "content2"}
        ]
        result = {"description": "test", "extracted_data": {}, "technical_details": {}}
        
        confidence = handler._calculate_confidence_score(processed_media, result)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_estimate_audio_duration(self, handler):
        """Test audio duration estimation"""
        audio_file = MediaFile(content=b"x" * 1000, mime_type="audio/mpeg")
        
        duration = handler._estimate_audio_duration(audio_file)
        
        assert duration > 0
        assert isinstance(duration, float)
    
    def test_determine_document_type(self, handler):
        """Test document type determination"""
        pdf_file = MediaFile(content=b"pdf", mime_type="application/pdf")
        assert handler._determine_document_type(pdf_file) == "pdf"
        
        word_file = MediaFile(content=b"doc", mime_type="application/msword")
        assert handler._determine_document_type(word_file) == "word"
        
        text_file = MediaFile(content=b"text", mime_type="text/plain")
        assert handler._determine_document_type(text_file) == "text"
        
        unknown_file = MediaFile(content=b"unknown", mime_type="application/unknown")
        assert handler._determine_document_type(unknown_file) == "unknown"
    
    @patch('multimodal_handler.Image')
    def test_analyze_image(self, mock_image, handler):
        """Test image analysis"""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.format = "JPEG"
        mock_img.mode = "RGB"
        mock_img.size = (100, 100)
        mock_img.width = 100
        mock_img.height = 100
        mock_image.open.return_value = mock_img
        
        image_file = MediaFile(content=b"fake_image", mime_type="image/jpeg")
        
        result = handler.analyze_image(image_file, AnalysisType.DETAILED)
        
        assert "basic_info" in result
        assert "content_analysis" in result
        assert "technical_analysis" in result
        assert result["basic_info"]["format"] == "JPEG"
        assert result["basic_info"]["width"] == 100
    
    def test_analyze_image_with_text_extraction(self, handler):
        """Test image analysis with text extraction"""
        with patch('multimodal_handler.Image') as mock_image:
            mock_img = Mock()
            mock_img.format = "JPEG"
            mock_img.mode = "RGB"
            mock_img.size = (100, 100)
            mock_img.width = 100
            mock_img.height = 100
            mock_image.open.return_value = mock_img
            
            image_file = MediaFile(content=b"fake_image", mime_type="image/jpeg")
            
            result = handler.analyze_image(image_file, AnalysisType.DETAILED, extract_text=True)
            
            assert "extracted_text" in result
            assert result["extracted_text"] is not None
    
    def test_analyze_image_error_handling(self, handler):
        """Test image analysis error handling"""
        # Invalid image data
        invalid_image = MediaFile(content=b"not_an_image", mime_type="image/jpeg")
        
        result = handler.analyze_image(invalid_image)
        
        assert "error" in result
    
    def test_analyze_document(self, handler, sample_text_file):
        """Test document analysis"""
        result = handler.analyze_document(sample_text_file)
        
        assert "document_type" in result
        assert "text_content" in result
        assert "metadata" in result
        assert "statistics" in result
        assert result["document_type"] == "text"
    
    def test_analyze_document_with_structure(self, handler, sample_text_file):
        """Test document analysis with structure extraction"""
        result = handler.analyze_document(sample_text_file, extract_structure=True)
        
        assert "structure_analysis" in result
        assert result["structure_analysis"] is not None
    
    def test_compare_media_insufficient_files(self, handler):
        """Test media comparison with insufficient files"""
        single_file = [MediaFile(content=b"test", mime_type="text/plain")]
        
        result = handler.compare_media(single_file)
        
        assert "error" in result
        assert "At least 2 media files required" in result["error"]
    
    def test_compare_media_success(self, handler, sample_text_file):
        """Test successful media comparison"""
        file1 = sample_text_file
        file2 = MediaFile(content=b"Different content", mime_type="text/plain", filename="test2.txt")
        
        result = handler.compare_media([file1, file2])
        
        assert "comparison_type" in result
        assert "individual_comparisons" in result
        assert "summary" in result
        assert len(result["individual_comparisons"]) == 1
    
    def test_extract_multimedia_content_default(self, handler, sample_text_file):
        """Test multimedia content extraction with default types"""
        result = handler.extract_multimedia_content([sample_text_file])
        
        assert len(result) == 1
        file_key = list(result.keys())[0]
        assert "text" in result[file_key]
        assert "metadata" in result[file_key]
        assert "technical_specs" in result[file_key]
    
    def test_extract_multimedia_content_custom_types(self, handler, sample_text_file):
        """Test multimedia content extraction with custom types"""
        result = handler.extract_multimedia_content([sample_text_file], ["text", "metadata"])
        
        file_key = list(result.keys())[0]
        assert "text" in result[file_key]
        assert "metadata" in result[file_key]
        assert "technical_specs" not in result[file_key]
    
    def test_process_multimodal_request_success(self, handler, sample_text_file):
        """Test successful multimodal request processing"""
        prompt = "Analyze this document"
        media_files = [sample_text_file]
        
        result = handler.process_multimodal_request(
            prompt, 
            media_files, 
            AnalysisType.DETAILED,
            ProcessingQuality.BALANCED
        )
        
        assert result.success
        assert result.content_description != ""
        assert result.processing_time > 0
        assert 0 <= result.confidence_score <= 1
        assert len(result.errors) == 0
    
    def test_process_multimodal_request_validation_failure(self, handler):
        """Test multimodal request with validation failure"""
        prompt = ""  # Empty prompt
        media_files = []  # No media files
        
        result = handler.process_multimodal_request(prompt, media_files)
        
        assert not result.success
        assert len(result.errors) > 0
        assert "Validation failed" in result.content_description
    
    def test_process_multimodal_request_with_custom_instructions(self, handler, sample_text_file):
        """Test multimodal request with custom instructions"""
        prompt = "Analyze this document"
        media_files = [sample_text_file]
        custom_instructions = "Focus on technical aspects"
        
        result = handler.process_multimodal_request(
            prompt, 
            media_files,
            custom_instructions=custom_instructions
        )
        
        assert result.success
    
    def test_compare_images_simulation(self, handler):
        """Test image comparison simulation"""
        file1 = MediaFile(content=b"image1", mime_type="image/jpeg", filename="img1.jpg")
        file2 = MediaFile(content=b"image2", mime_type="image/jpeg", filename="img2.jpg")
        
        result = handler._compare_images(file1, file2, "similarity")
        
        assert "similarity_score" in result
        assert "visual_differences" in result
        assert "format_comparison" in result
        assert 0 <= result["similarity_score"] <= 1
    
    def test_compare_documents_simulation(self, handler):
        """Test document comparison simulation"""
        file1 = MediaFile(content=b"doc1", mime_type="text/plain", filename="doc1.txt")
        file2 = MediaFile(content=b"doc2", mime_type="text/plain", filename="doc2.txt")
        
        result = handler._compare_documents(file1, file2, "content")
        
        assert "content_similarity" in result
        assert "structural_similarity" in result
        assert "format_comparison" in result
        assert 0 <= result["content_similarity"] <= 1
    
    def test_generate_thumbnails(self, handler):
        """Test thumbnail generation"""
        with patch('multimodal_handler.Image') as mock_image:
            mock_img = Mock()
            mock_img.copy.return_value = mock_img
            mock_img.save = Mock()
            mock_image.open.return_value = mock_img
            
            image_file = MediaFile(content=b"fake_image", mime_type="image/jpeg")
            
            thumbnails = handler._generate_thumbnails(image_file)
            
            assert "small" in thumbnails
            assert "medium" in thumbnails
            assert "large" in thumbnails
    
    def test_generate_thumbnails_non_image(self, handler):
        """Test thumbnail generation for non-image file"""
        text_file = MediaFile(content=b"text", mime_type="text/plain")
        
        thumbnails = handler._generate_thumbnails(text_file)
        
        assert len(thumbnails) == 0
    
    def test_extract_text_content_image(self, handler):
        """Test text extraction from image"""
        with patch('multimodal_handler.Image') as mock_image:
            mock_img = Mock()
            mock_image.open.return_value = mock_img
            
            image_file = MediaFile(content=b"fake_image", mime_type="image/jpeg")
            
            text = handler._extract_text_content(image_file)
            
            assert isinstance(text, str)
    
    def test_extract_text_content_document(self, handler):
        """Test text extraction from document"""
        text_file = MediaFile(content=b"text content", mime_type="text/plain")
        
        text = handler._extract_text_content(text_file)
        
        assert isinstance(text, str)
    
    def test_extract_text_content_unsupported(self, handler):
        """Test text extraction from unsupported file"""
        audio_file = MediaFile(content=b"audio", mime_type="audio/mpeg")
        
        text = handler._extract_text_content(audio_file)
        
        assert text == ""

class TestMultimodalIntegration:
    """Integration tests for multimodal processing"""
    
    def test_complete_image_processing_flow(self):
        """Test complete image processing workflow"""
        handler = GeminiMultimodalHandler()
        
        # Create test image
        with patch('multimodal_handler.Image') as mock_image:
            mock_img = Mock()
            mock_img.format = "JPEG"
            mock_img.mode = "RGB" 
            mock_img.size = (200, 200)
            mock_img.width = 200
            mock_img.height = 200
            mock_img.save = Mock()
            mock_image.open.return_value = mock_img
            
            image_file = MediaFile(
                content=b"fake_image_content",
                mime_type="image/jpeg",
                filename="test_image.jpg"
            )
            
            # Process the image
            result = handler.process_multimodal_request(
                "Analyze this image in detail",
                [image_file],
                AnalysisType.DETAILED,
                ProcessingQuality.HIGH
            )
            
            assert result.success
            assert result.confidence_score > 0
            assert len(result.warnings) == 0
    
    def test_complete_document_processing_flow(self):
        """Test complete document processing workflow"""
        handler = GeminiMultimodalHandler()
        
        document_file = MediaFile(
            content=b"This is a test document with important content to analyze.",
            mime_type="text/plain",
            filename="test_document.txt"
        )
        
        result = handler.process_multimodal_request(
            "Extract key information from this document",
            [document_file],
            AnalysisType.CONTENT_EXTRACTION,
            ProcessingQuality.HIGH
        )
        
        assert result.success
        assert result.confidence_score > 0
        assert "extraction_summary" in result.extracted_data
    
    def test_mixed_media_processing(self):
        """Test processing multiple different media types"""
        handler = GeminiMultimodalHandler()
        
        # Create mixed media files
        text_file = MediaFile(content=b"Document content", mime_type="text/plain", filename="doc.txt")
        
        with patch('multimodal_handler.Image') as mock_image:
            mock_img = Mock()
            mock_img.format = "JPEG"
            mock_img.mode = "RGB"
            mock_img.size = (100, 100)
            mock_img.width = 100
            mock_img.height = 100
            mock_img.save = Mock()
            mock_image.open.return_value = mock_img
            
            image_file = MediaFile(content=b"image_content", mime_type="image/jpeg", filename="img.jpg")
            
            result = handler.process_multimodal_request(
                "Analyze both the document and image",
                [text_file, image_file],
                AnalysisType.DETAILED
            )
            
            assert result.success
            assert len(result.extracted_data.get("media_insights", [])) == 2

class TestMultimodalErrorHandling:
    """Test error handling in multimodal processing"""
    
    def test_processing_with_corrupted_image(self):
        """Test handling of corrupted image files"""
        handler = GeminiMultimodalHandler()
        
        # Corrupted image file
        corrupted_image = MediaFile(
            content=b"not_an_image_at_all",
            mime_type="image/jpeg",
            filename="corrupted.jpg"
        )
        
        result = handler.process_multimodal_request(
            "Analyze this image",
            [corrupted_image]
        )
        
        # Should handle gracefully
        assert len(result.warnings) > 0 or not result.success
    
    def test_processing_with_empty_file(self):
        """Test handling of empty files"""
        handler = GeminiMultimodalHandler()
        
        empty_file = MediaFile(
            content=b"",
            mime_type="text/plain",
            filename="empty.txt"
        )
        
        result = handler.process_multimodal_request(
            "Analyze this file",
            [empty_file]
        )
        
        assert not result.success
        assert "no content" in " ".join(result.errors).lower()

class TestPerformance:
    """Performance tests for multimodal processing"""
    
    def test_large_file_processing_simulation(self):
        """Test processing of large files (simulated)"""
        handler = GeminiMultimodalHandler()
        
        # Large text file (within limits)
        large_content = b"Large content " * 1000  # ~13KB
        large_file = MediaFile(
            content=large_content,
            mime_type="text/plain",
            filename="large.txt"
        )
        
        import time
        start_time = time.time()
        
        result = handler.process_multimodal_request(
            "Analyze this large document",
            [large_file],
            ProcessingQuality.FAST  # Use fast processing
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert result.success
        assert processing_time < 5.0  # Should complete within 5 seconds
    
    def test_multiple_files_processing(self):
        """Test processing multiple files efficiently"""
        handler = GeminiMultimodalHandler()
        
        # Create multiple small files
        files = []
        for i in range(5):
            file = MediaFile(
                content=f"Content of file {i}".encode(),
                mime_type="text/plain",
                filename=f"file_{i}.txt"
            )
            files.append(file)
        
        result = handler.process_multimodal_request(
            "Analyze all these files",
            files,
            ProcessingQuality.FAST
        )
        
        assert result.success
        assert len(result.extracted_data.get("media_insights", [])) == 5

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
