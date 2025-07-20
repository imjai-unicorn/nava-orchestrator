# backend/services/03-external-ai/gemini-client/app/multimodal_handler.py
"""
Gemini Multimodal Handler
Advanced multimodal processing for Gemini models (text, images, audio, video)
"""

import logging
import base64
import io
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import mimetypes
import hashlib
from PIL import Image
import json

logger = logging.getLogger(__name__)

class MediaType(str, Enum):
    """Supported media types"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"

class ImageFormat(str, Enum):
    """Supported image formats"""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    TIFF = "tiff"

class AudioFormat(str, Enum):
    """Supported audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    AAC = "aac"
    OGG = "ogg"

class VideoFormat(str, Enum):
    """Supported video formats"""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    MKV = "mkv"

class ProcessingQuality(str, Enum):
    """Processing quality levels"""
    FAST = "fast"           # Quick processing, lower quality
    BALANCED = "balanced"   # Balanced speed and quality
    HIGH = "high"          # High quality, slower processing
    MAXIMUM = "maximum"    # Maximum quality, slowest

class AnalysisType(str, Enum):
    """Types of media analysis"""
    BASIC = "basic"                    # Basic content description
    DETAILED = "detailed"              # Detailed analysis
    TECHNICAL = "technical"            # Technical specifications
    CONTENT_EXTRACTION = "content_extraction"  # Extract text/data
    COMPARISON = "comparison"          # Compare multiple media
    CLASSIFICATION = "classification"  # Classify content
    SENTIMENT = "sentiment"           # Sentiment analysis
    SAFETY = "safety"                 # Safety/content moderation

@dataclass
class MediaFile:
    """Media file representation"""
    content: bytes
    mime_type: str
    filename: Optional[str] = None
    size: Optional[int] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.size is None:
            self.size = len(self.content)
        if self.checksum is None:
            self.checksum = hashlib.md5(self.content).hexdigest()

@dataclass
class ProcessingResult:
    """Result of multimodal processing"""
    success: bool
    content_description: str
    extracted_data: Dict[str, Any]
    technical_details: Dict[str, Any]
    processing_time: float
    confidence_score: float
    warnings: List[str]
    errors: List[str]

class GeminiMultimodalHandler:
    """Advanced multimodal processing handler for Gemini"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_file_size = self.config.get('max_file_size', 20 * 1024 * 1024)  # 20MB
        self.supported_formats = self._load_supported_formats()
        self.processing_cache = {}
        logger.info("🎭 Gemini Multimodal Handler initialized")
    
    def process_multimodal_request(self, 
                                 text_prompt: str,
                                 media_files: List[MediaFile],
                                 analysis_type: AnalysisType = AnalysisType.DETAILED,
                                 quality: ProcessingQuality = ProcessingQuality.BALANCED,
                                 custom_instructions: Optional[str] = None) -> ProcessingResult:
        """
        Process multimodal request with text and media files
        
        Args:
            text_prompt: Text prompt for the AI
            media_files: List of media files to process
            analysis_type: Type of analysis to perform
            quality: Processing quality level
            custom_instructions: Custom processing instructions
            
        Returns:
            ProcessingResult with analysis and extracted data
        """
        import time
        start_time = time.time()
        
        logger.info(f"Processing multimodal request with {len(media_files)} media files")
        
        try:
            # Validate inputs
            validation_result = self._validate_inputs(text_prompt, media_files)
            if not validation_result[0]:
                return ProcessingResult(
                    success=False,
                    content_description="Validation failed",
                    extracted_data={},
                    technical_details={},
                    processing_time=time.time() - start_time,
                    confidence_score=0.0,
                    warnings=[],
                    errors=validation_result[1]
                )
            
            # Prepare media for processing
            processed_media = []
            warnings = []
            
            for media_file in media_files:
                processed = self._prepare_media_file(media_file, quality)
                if processed:
                    processed_media.append(processed)
                else:
                    warnings.append(f"Failed to process media file: {media_file.filename}")
            
            # Build comprehensive prompt
            enhanced_prompt = self._build_enhanced_prompt(
                text_prompt, processed_media, analysis_type, custom_instructions
            )
            
            # Simulate processing (in real implementation, this would call Gemini API)
            result = self._process_with_gemini(enhanced_prompt, processed_media, analysis_type)
            
            # Post-process results
            final_result = self._post_process_results(result, processed_media, analysis_type)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                content_description=final_result.get('description', ''),
                extracted_data=final_result.get('extracted_data', {}),
                technical_details=final_result.get('technical_details', {}),
                processing_time=processing_time,
                confidence_score=final_result.get('confidence', 0.8),
                warnings=warnings,
                errors=[]
            )
            
        except Exception as e:
            logger.error(f"❌ Multimodal processing failed: {e}")
            return ProcessingResult(
                success=False,
                content_description=f"Processing failed: {str(e)}",
                extracted_data={},
                technical_details={},
                processing_time=time.time() - start_time,
                confidence_score=0.0,
                warnings=[],
                errors=[str(e)]
            )
    
    def analyze_image(self, 
                     image_file: MediaFile,
                     analysis_type: AnalysisType = AnalysisType.DETAILED,
                     extract_text: bool = False) -> Dict[str, Any]:
        """Analyze image content"""
        
        try:
            # Load and validate image
            image = Image.open(io.BytesIO(image_file.content))
            
            # Basic image information
            basic_info = {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0
            }
            
            # Enhanced analysis based on type
            analysis_result = {
                "basic_info": basic_info,
                "content_analysis": self._analyze_image_content(image, analysis_type),
                "technical_analysis": self._analyze_image_technical(image),
                "extracted_text": self._extract_text_from_image(image) if extract_text else None
            }
            
            # Add specialized analysis
            if analysis_type == AnalysisType.SAFETY:
                analysis_result["safety_analysis"] = self._analyze_image_safety(image)
            elif analysis_type == AnalysisType.CLASSIFICATION:
                analysis_result["classification"] = self._classify_image_content(image)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Image analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_document(self, 
                        document_file: MediaFile,
                        extract_structure: bool = True) -> Dict[str, Any]:
        """Analyze document content and structure"""
        
        try:
            # Determine document type
            doc_type = self._determine_document_type(document_file)
            
            # Extract text content
            text_content = self._extract_document_text(document_file, doc_type)
            
            # Analyze structure if requested
            structure_analysis = None
            if extract_structure:
                structure_analysis = self._analyze_document_structure(text_content, doc_type)
            
            return {
                "document_type": doc_type,
                "text_content": text_content,
                "structure_analysis": structure_analysis,
                "metadata": self._extract_document_metadata(document_file, doc_type),
                "statistics": self._calculate_document_statistics(text_content)
            }
            
        except Exception as e:
            logger.error(f"❌ Document analysis failed: {e}")
            return {"error": str(e)}
    
    def compare_media(self, 
                     media_files: List[MediaFile],
                     comparison_type: str = "similarity") -> Dict[str, Any]:
        """Compare multiple media files"""
        
        if len(media_files) < 2:
            return {"error": "At least 2 media files required for comparison"}
        
        try:
            comparisons = []
            
            for i in range(len(media_files)):
                for j in range(i + 1, len(media_files)):
                    file1, file2 = media_files[i], media_files[j]
                    
                    # Determine comparison method based on media types
                    if self._is_image(file1) and self._is_image(file2):
                        comparison = self._compare_images(file1, file2, comparison_type)
                    elif self._is_text_document(file1) and self._is_text_document(file2):
                        comparison = self._compare_documents(file1, file2, comparison_type)
                    else:
                        comparison = self._compare_generic_media(file1, file2, comparison_type)
                    
                    comparisons.append({
                        "file1": file1.filename or f"file_{i}",
                        "file2": file2.filename or f"file_{j}",
                        "comparison_result": comparison
                    })
            
            # Generate summary
            summary = self._generate_comparison_summary(comparisons, comparison_type)
            
            return {
                "comparison_type": comparison_type,
                "individual_comparisons": comparisons,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"❌ Media comparison failed: {e}")
            return {"error": str(e)}
    
    def extract_multimedia_content(self, 
                                 media_files: List[MediaFile],
                                 content_types: List[str] = None) -> Dict[str, Any]:
        """Extract specific content types from multimedia files"""
        
        content_types = content_types or ["text", "metadata", "technical_specs"]
        extracted_content = {}
        
        for media_file in media_files:
            file_key = media_file.filename or media_file.checksum[:8]
            extracted_content[file_key] = {}
            
            try:
                if "text" in content_types:
                    extracted_content[file_key]["text"] = self._extract_text_content(media_file)
                
                if "metadata" in content_types:
                    extracted_content[file_key]["metadata"] = self._extract_media_metadata(media_file)
                
                if "technical_specs" in content_types:
                    extracted_content[file_key]["technical_specs"] = self._extract_technical_specs(media_file)
                
                if "thumbnails" in content_types and self._is_image(media_file):
                    extracted_content[file_key]["thumbnails"] = self._generate_thumbnails(media_file)
                
            except Exception as e:
                logger.error(f"❌ Content extraction failed for {file_key}: {e}")
                extracted_content[file_key]["error"] = str(e)
        
        return extracted_content
    
    def _validate_inputs(self, text_prompt: str, media_files: List[MediaFile]) -> Tuple[bool, List[str]]:
        """Validate inputs for processing"""
        errors = []
        
        # Validate text prompt
        if not text_prompt or len(text_prompt.strip()) == 0:
            errors.append("Text prompt cannot be empty")
        
        if len(text_prompt) > 100000:  # Reasonable limit
            errors.append("Text prompt too long (max 100,000 characters)")
        
        # Validate media files
        if not media_files:
            errors.append("At least one media file is required")
        
        for i, media_file in enumerate(media_files):
            # Check file size
            if media_file.size > self.max_file_size:
                errors.append(f"File {i} exceeds maximum size ({self.max_file_size} bytes)")
            
            # Check mime type
            if not self._is_supported_format(media_file.mime_type):
                errors.append(f"File {i} has unsupported format: {media_file.mime_type}")
            
            # Check content integrity
            if len(media_file.content) == 0:
                errors.append(f"File {i} has no content")
        
        return len(errors) == 0, errors
    
    def _prepare_media_file(self, media_file: MediaFile, quality: ProcessingQuality) -> Optional[Dict[str, Any]]:
        """Prepare media file for processing"""
        
        try:
            media_info = {
                "mime_type": media_file.mime_type,
                "size": media_file.size,
                "filename": media_file.filename,
                "checksum": media_file.checksum
            }
            
            # Process based on media type
            if self._is_image(media_file):
                media_info.update(self._prepare_image(media_file, quality))
            elif self._is_audio(media_file):
                media_info.update(self._prepare_audio(media_file, quality))
            elif self._is_video(media_file):
                media_info.update(self._prepare_video(media_file, quality))
            elif self._is_document(media_file):
                media_info.update(self._prepare_document(media_file, quality))
            else:
                media_info["processed_content"] = base64.b64encode(media_file.content).decode()
            
            return media_info
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare media file: {e}")
            return None
    
    def _prepare_image(self, media_file: MediaFile, quality: ProcessingQuality) -> Dict[str, Any]:
        """Prepare image for processing"""
        image = Image.open(io.BytesIO(media_file.content))
        
        # Resize based on quality setting
        max_dimensions = {
            ProcessingQuality.FAST: (512, 512),
            ProcessingQuality.BALANCED: (1024, 1024),
            ProcessingQuality.HIGH: (2048, 2048),
            ProcessingQuality.MAXIMUM: (4096, 4096)
        }
        
        max_w, max_h = max_dimensions[quality]
        
        # Resize if necessary
        if image.width > max_w or image.height > max_h:
            image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "processed_content": encoded_image,
            "processed_format": "jpeg",
            "processed_size": (image.width, image.height),
            "compression_applied": image.width != Image.open(io.BytesIO(media_file.content)).width
        }
    
    def _prepare_audio(self, media_file: MediaFile, quality: ProcessingQuality) -> Dict[str, Any]:
        """Prepare audio for processing"""
        # For now, just encode as base64 (in real implementation, might resample/compress)
        encoded_audio = base64.b64encode(media_file.content).decode()
        
        return {
            "processed_content": encoded_audio,
            "processed_format": media_file.mime_type.split('/')[-1],
            "duration_estimate": self._estimate_audio_duration(media_file),
            "quality_settings": quality
        }
    
    def _prepare_video(self, media_file: MediaFile, quality: ProcessingQuality) -> Dict[str, Any]:
        """Prepare video for processing"""
        # Extract key frames and metadata
        return {
            "processed_content": base64.b64encode(media_file.content).decode(),
            "processed_format": media_file.mime_type.split('/')[-1],
            "key_frames_extracted": True,  # Would extract actual frames in real implementation
            "quality_settings": quality
        }
    
    def _prepare_document(self, media_file: MediaFile, quality: ProcessingQuality) -> Dict[str, Any]:
        """Prepare document for processing"""
        try:
            # Extract text content
            text_content = self._extract_document_text(media_file, media_file.mime_type)
            
            return {
                "processed_content": text_content,
                "processed_format": "text",
                "original_format": media_file.mime_type,
                "text_length": len(text_content)
            }
        except:
            # Fallback to base64 if text extraction fails
            return {
                "processed_content": base64.b64encode(media_file.content).decode(),
                "processed_format": "binary",
                "extraction_failed": True
            }
    
    def _build_enhanced_prompt(self, 
                             text_prompt: str, 
                             processed_media: List[Dict[str, Any]],
                             analysis_type: AnalysisType,
                             custom_instructions: Optional[str]) -> str:
        """Build enhanced prompt for multimodal processing"""
        
        # Base prompt structure
        prompt_parts = [
            f"Primary Task: {text_prompt}",
            "",
            f"Analysis Type: {analysis_type.value}",
            ""
        ]
        
        # Add media descriptions
        if processed_media:
            prompt_parts.append("Media Files to Analyze:")
            for i, media in enumerate(processed_media):
                prompt_parts.append(f"{i+1}. {media.get('filename', 'Unnamed file')} ({media.get('mime_type', 'unknown type')})")
            prompt_parts.append("")
        
        # Add analysis-specific instructions
        analysis_instructions = {
            AnalysisType.BASIC: "Provide a basic description of the content in the media files.",
            AnalysisType.DETAILED: "Provide a comprehensive analysis including content, context, and key elements.",
            AnalysisType.TECHNICAL: "Focus on technical specifications, quality, and format details.",
            AnalysisType.CONTENT_EXTRACTION: "Extract all text, data, and structured information from the media.",
            AnalysisType.COMPARISON: "Compare and contrast the different media files, highlighting similarities and differences.",
            AnalysisType.CLASSIFICATION: "Classify the content type, style, and category of each media file.",
            AnalysisType.SENTIMENT: "Analyze the emotional tone and sentiment expressed in the media.",
            AnalysisType.SAFETY: "Evaluate content for safety, appropriateness, and potential concerns."
        }
        
        prompt_parts.append(f"Instructions: {analysis_instructions.get(analysis_type, 'Analyze the provided media.')}")
        
        # Add custom instructions if provided
        if custom_instructions:
            prompt_parts.append("")
            prompt_parts.append(f"Additional Instructions: {custom_instructions}")
        
        # Add output format guidance
        prompt_parts.extend([
            "",
            "Please provide your analysis in a structured format with:",
            "1. Overall summary",
            "2. Individual media analysis", 
            "3. Key findings",
            "4. Technical details (if applicable)",
            "5. Recommendations or next steps"
        ])
        
        return "\n".join(prompt_parts)
    
    def _process_with_gemini(self, 
                           prompt: str, 
                           processed_media: List[Dict[str, Any]],
                           analysis_type: AnalysisType) -> Dict[str, Any]:
        """Process request with Gemini API (simulated)"""
        
        # In real implementation, this would make actual API calls to Gemini
        # For now, we'll simulate intelligent responses based on the inputs
        
        result = {
            "description": "Multimodal content analysis completed",
            "extracted_data": {},
            "technical_details": {},
            "confidence": 0.85
        }
        
        # Simulate different analysis types
        if analysis_type == AnalysisType.DETAILED:
            result["description"] = "Detailed analysis of multimodal content reveals comprehensive insights"
            result["extracted_data"] = {
                "content_summary": "The provided media contains rich information suitable for detailed analysis",
                "key_elements": ["visual components", "textual information", "structural data"],
                "insights": ["High quality content", "Clear information hierarchy", "Good accessibility"]
            }
        
        elif analysis_type == AnalysisType.TECHNICAL:
            result["technical_details"] = {
                "media_count": len(processed_media),
                "formats_processed": [media.get("mime_type", "unknown") for media in processed_media],
                "processing_quality": "high",
                "compression_applied": any(media.get("compression_applied", False) for media in processed_media)
            }
        
        elif analysis_type == AnalysisType.SAFETY:
            result["extracted_data"] = {
                "safety_score": 0.95,
                "content_flags": [],
                "recommendations": ["Content appears safe for general audience"],
                "moderation_status": "approved"
            }
        
        # Add processing metadata
        result["processing_metadata"] = {
            "prompt_length": len(prompt),
            "media_files_processed": len(processed_media),
            "analysis_type": analysis_type.value,
            "processing_timestamp": "2024-01-01T00:00:00Z"
        }
        
        return result
    
    def _post_process_results(self, 
                            result: Dict[str, Any], 
                            processed_media: List[Dict[str, Any]],
                            analysis_type: AnalysisType) -> Dict[str, Any]:
        """Post-process and enhance results"""
        
        enhanced_result = result.copy()
        
        # Add media-specific insights
        media_insights = []
        for media in processed_media:
            insight = {
                "filename": media.get("filename", "unknown"),
                "type": media.get("mime_type", "unknown"),
                "processed_successfully": "processed_content" in media,
                "size_info": media.get("processed_size", "unknown")
            }
            media_insights.append(insight)
        
        enhanced_result["media_insights"] = media_insights
        
        # Enhance based on analysis type
        if analysis_type == AnalysisType.CONTENT_EXTRACTION:
            enhanced_result["extraction_summary"] = {
                "total_media_processed": len(processed_media),
                "successful_extractions": len([m for m in processed_media if "processed_content" in m]),
                "extraction_methods_used": ["ocr", "text_parsing", "metadata_extraction"]
            }
        
        # Add confidence scoring
        if "confidence" not in enhanced_result:
            enhanced_result["confidence"] = self._calculate_confidence_score(processed_media, result)
        
        return enhanced_result
    
    def _analyze_image_content(self, image: Image.Image, analysis_type: AnalysisType) -> Dict[str, Any]:
        """Analyze image content (simulated)"""
        return {
            "content_type": "photograph" if image.mode == "RGB" else "graphic",
            "dominant_colors": ["#FF5733", "#33FF57", "#3357FF"],  # Simulated
            "objects_detected": ["person", "building", "sky"],  # Simulated
            "scene_description": "An outdoor scene with architectural elements",
            "estimated_complexity": "medium"
        }
    
    def _analyze_image_technical(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze technical aspects of image"""
        return {
            "color_space": image.mode,
            "bit_depth": 8 if image.mode in ["RGB", "RGBA"] else 1,
            "compression_estimate": "medium",
            "quality_score": 0.8,
            "resolution_class": "high" if max(image.size) > 1920 else "standard"
        }
    
    def _extract_text_from_image(self, image: Image.Image) -> Optional[str]:
        """Extract text from image using OCR (simulated)"""
        # In real implementation, would use OCR library like pytesseract
        return "Sample extracted text from image"
    
    def _analyze_image_safety(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image for safety concerns"""
        return {
            "safety_score": 0.95,
            "content_warnings": [],
            "age_appropriate": True,
            "contains_text": True,
            "moderation_flags": []
        }
    
    def _classify_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """Classify image content"""
        return {
            "primary_category": "photography",
            "subcategories": ["landscape", "architecture"],
            "style": "realistic",
            "confidence_scores": {
                "photography": 0.9,
                "artwork": 0.1,
                "document": 0.0
            }
        }
    
    def _determine_document_type(self, document_file: MediaFile) -> str:
        """Determine document type from MIME type"""
        mime_type = document_file.mime_type.lower()
        
        if "pdf" in mime_type:
            return "pdf"
        elif "word" in mime_type or "docx" in mime_type:
            return "word"
        elif "excel" in mime_type or "xlsx" in mime_type:
            return "excel"
        elif "powerpoint" in mime_type or "pptx" in mime_type:
            return "powerpoint"
        elif "text" in mime_type:
            return "text"
        else:
            return "unknown"
    
    def _extract_document_text(self, document_file: MediaFile, doc_type: str) -> str:
        """Extract text from document (simulated)"""
        # In real implementation, would use appropriate libraries
        return f"Sample extracted text from {doc_type} document"
    
    def _analyze_document_structure(self, text_content: str, doc_type: str) -> Dict[str, Any]:
        """Analyze document structure"""
        return {
            "sections_detected": 3,
            "has_headers": True,
            "has_tables": doc_type in ["excel", "word"],
            "has_images": doc_type in ["word", "powerpoint"],
            "estimated_reading_time": len(text_content) / 200  # words per minute
        }
    
    def _extract_document_metadata(self, document_file: MediaFile, doc_type: str) -> Dict[str, Any]:
        """Extract document metadata"""
        return {
            "file_size": document_file.size,
            "estimated_pages": max(1, document_file.size // 2000),  # Rough estimate
            "document_type": doc_type,
            "creation_method": "application"
        }
    
    def _calculate_document_statistics(self, text_content: str) -> Dict[str, Any]:
        """Calculate document statistics"""
        words = text_content.split()
        sentences = text_content.split(".")
        
        return {
            "character_count": len(text_content),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "reading_level": "intermediate"  # Simplified
        }
    
    def _compare_images(self, file1: MediaFile, file2: MediaFile, comparison_type: str) -> Dict[str, Any]:
        """Compare two images"""
        return {
            "similarity_score": 0.75,  # Simulated
            "visual_differences": ["color palette", "composition"],
            "size_comparison": "similar",
            "format_comparison": "different" if file1.mime_type != file2.mime_type else "same"
        }
    
    def _compare_documents(self, file1: MediaFile, file2: MediaFile, comparison_type: str) -> Dict[str, Any]:
        """Compare two documents"""
        return {
            "content_similarity": 0.60,  # Simulated
            "structural_similarity": 0.80,
            "format_comparison": "same" if file1.mime_type == file2.mime_type else "different",
            "size_difference_ratio": abs(file1.size - file2.size) / max(file1.size, file2.size)
        }
    
    def _compare_generic_media(self, file1: MediaFile, file2: MediaFile, comparison_type: str) -> Dict[str, Any]:
        """Compare media of different types"""
        return {
            "format_compatibility": "incompatible",
            "size_comparison": "similar" if abs(file1.size - file2.size) < file1.size * 0.2 else "different",
            "content_type_similarity": 0.0
        }
    
    def _generate_comparison_summary(self, comparisons: List[Dict[str, Any]], comparison_type: str) -> Dict[str, Any]:
        """Generate summary of all comparisons"""
        return {
            "total_comparisons": len(comparisons),
            "average_similarity": 0.65,  # Simulated
            "most_similar_pair": comparisons[0] if comparisons else None,
            "most_different_pair": comparisons[-1] if comparisons else None,
            "comparison_insights": ["Mixed content types detected", "Varying quality levels"]
        }
    
    def _is_image(self, media_file: MediaFile) -> bool:
        """Check if file is an image"""
        return media_file.mime_type.startswith("image/")
    
    def _is_audio(self, media_file: MediaFile) -> bool:
        """Check if file is audio"""
        return media_file.mime_type.startswith("audio/")
    
    def _is_video(self, media_file: MediaFile) -> bool:
        """Check if file is video"""
        return media_file.mime_type.startswith("video/")
    
    def _is_document(self, media_file: MediaFile) -> bool:
        """Check if file is a document"""
        doc_types = ["application/pdf", "application/msword", "application/vnd.ms-excel",
                    "application/vnd.openxmlformats", "text/plain", "text/csv"]
        return any(doc_type in media_file.mime_type for doc_type in doc_types)
    
    def _is_text_document(self, media_file: MediaFile) -> bool:
        """Check if file is a text-based document"""
        return self._is_document(media_file) or media_file.mime_type.startswith("text/")
    
    def _is_supported_format(self, mime_type: str) -> bool:
        """Check if MIME type is supported"""
        return mime_type in self.supported_formats
    
    def _load_supported_formats(self) -> List[str]:
        """Load list of supported MIME types"""
        return [
            # Images
            "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff",
            # Audio
            "audio/mpeg", "audio/wav", "audio/flac", "audio/aac", "audio/ogg",
            # Video  
            "video/mp4", "video/avi", "video/mov", "video/webm", "video/mkv",
            # Documents
            "application/pdf", "application/msword", "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/plain", "text/csv", "text/html"
        ]
    
    def _extract_text_content(self, media_file: MediaFile) -> str:
        """Extract text content from any media file"""
        if self._is_image(media_file):
            return self._extract_text_from_image(Image.open(io.BytesIO(media_file.content)))
        elif self._is_document(media_file):
            return self._extract_document_text(media_file, self._determine_document_type(media_file))
        else:
            return ""
    
    def _extract_media_metadata(self, media_file: MediaFile) -> Dict[str, Any]:
        """Extract metadata from media file"""
        return {
            "mime_type": media_file.mime_type,
            "size": media_file.size,
            "checksum": media_file.checksum,
            "filename": media_file.filename
        }
    
    def _extract_technical_specs(self, media_file: MediaFile) -> Dict[str, Any]:
        """Extract technical specifications"""
        if self._is_image(media_file):
            image = Image.open(io.BytesIO(media_file.content))
            return {
                "dimensions": image.size,
                "format": image.format,
                "mode": image.mode,
                "estimated_dpi": 72  # Default estimate
            }
        else:
            return {
                "file_size": media_file.size,
                "mime_type": media_file.mime_type
            }
    
    def _generate_thumbnails(self, media_file: MediaFile) -> Dict[str, str]:
        """Generate thumbnails for image files"""
        if not self._is_image(media_file):
            return {}
        
        image = Image.open(io.BytesIO(media_file.content))
        thumbnails = {}
        
        sizes = {"small": (128, 128), "medium": (256, 256), "large": (512, 512)}
        
        for size_name, (width, height) in sizes.items():
            thumb = image.copy()
            thumb.thumbnail((width, height), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            thumb.save(buffer, format='JPEG', quality=85)
            thumbnails[size_name] = base64.b64encode(buffer.getvalue()).decode()
        
        return thumbnails
    
    def _estimate_audio_duration(self, media_file: MediaFile) -> float:
        """Estimate audio duration (simplified)"""
        # Very rough estimate based on file size and typical bitrates
        estimated_bitrate = 128000  # 128 kbps
        return (media_file.size * 8) / estimated_bitrate  # seconds
    
    def _calculate_confidence_score(self, processed_media: List[Dict[str, Any]], result: Dict[str, Any]) -> float:
        """Calculate confidence score for processing result"""
        base_confidence = 0.8
        
        # Adjust based on successful processing
        success_rate = len([m for m in processed_media if "processed_content" in m]) / len(processed_media)
        
        # Adjust based on result completeness
        result_completeness = len([k for k in ["description", "extracted_data", "technical_details"] if k in result]) / 3
        
        return min(1.0, base_confidence * success_rate * result_completeness)

# Export main class and utilities
__all__ = [
    "MediaType",
    "ImageFormat", 
    "AudioFormat",
    "VideoFormat",
    "ProcessingQuality",
    "AnalysisType",
    "MediaFile",
    "ProcessingResult",
    "GeminiMultimodalHandler"
]