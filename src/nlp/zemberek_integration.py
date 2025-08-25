"""
Zemberek-NLP Integration for Turkish Language Processing
TEKNOFEST 2025 - Advanced Turkish NLP with Zemberek
"""

import os
import sys
import json
import subprocess
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum

# For Java integration
try:
    import jpype
    import jpype.imports
    from jpype import JClass, JString
    JPYPE_AVAILABLE = True
except ImportError:
    JPYPE_AVAILABLE = False
    print("JPype not available. Using REST API mode for Zemberek.")

# For REST API fallback
import requests


logger = logging.getLogger(__name__)


class ZemberekMode(Enum):
    """Zemberek integration modes"""
    JAVA_DIRECT = "java_direct"  # Direct Java integration via JPype
    REST_API = "rest_api"        # REST API server
    SUBPROCESS = "subprocess"     # Command line interface


@dataclass
class ZemberekConfig:
    """Configuration for Zemberek integration"""
    mode: ZemberekMode = ZemberekMode.REST_API
    java_home: Optional[str] = None
    zemberek_jar_path: str = "lib/zemberek-full.jar"
    api_url: str = "http://localhost:4567"
    api_timeout: int = 30
    cache_enabled: bool = True
    cache_size: int = 10000
    log_level: str = "INFO"


@dataclass 
class MorphAnalysis:
    """Morphological analysis result"""
    surface: str
    lemma: str
    pos: str
    morphemes: List[str]
    formatted: str
    stem: str
    endings: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentenceAnalysis:
    """Sentence-level analysis result"""
    text: str
    tokens: List[str]
    morphology: List[MorphAnalysis]
    dependencies: Optional[List[Dict]] = None
    disambiguation: Optional[List[str]] = None


class ZemberekIntegration:
    """Zemberek-NLP integration for advanced Turkish NLP"""
    
    def __init__(self, config: Optional[ZemberekConfig] = None):
        self.config = config or ZemberekConfig()
        self.mode = self.config.mode
        self._cache = {} if self.config.cache_enabled else None
        
        # Initialize based on mode
        if self.mode == ZemberekMode.JAVA_DIRECT:
            self._init_java_mode()
        elif self.mode == ZemberekMode.REST_API:
            self._init_api_mode()
        elif self.mode == ZemberekMode.SUBPROCESS:
            self._init_subprocess_mode()
            
        logger.info(f"Zemberek initialized in {self.mode.value} mode")
        
    def _init_java_mode(self):
        """Initialize direct Java integration"""
        if not JPYPE_AVAILABLE:
            logger.warning("JPype not available, falling back to REST API mode")
            self.mode = ZemberekMode.REST_API
            self._init_api_mode()
            return
            
        try:
            # Start JVM if not started
            if not jpype.isJVMStarted():
                jpype.startJVM(
                    jpype.getDefaultJVMPath(),
                    f"-Djava.class.path={self.config.zemberek_jar_path}",
                    convertStrings=False
                )
                
            # Import Zemberek classes
            from zemberek.morphology import TurkishMorphology
            from zemberek.normalization import TurkishSpellChecker
            from zemberek.tokenization import TurkishTokenizer
            
            # Initialize components
            self.morphology = TurkishMorphology.createWithDefaults()
            self.tokenizer = TurkishTokenizer.DEFAULT
            self.spell_checker = TurkishSpellChecker(self.morphology)
            
            logger.info("Zemberek Java components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Java mode: {e}")
            self.mode = ZemberekMode.REST_API
            self._init_api_mode()
            
    def _init_api_mode(self):
        """Initialize REST API mode"""
        self.api_url = self.config.api_url
        
        # Test connection
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code != 200:
                logger.warning("Zemberek API server not responding, starting local server...")
                self._start_api_server()
        except requests.exceptions.RequestException:
            logger.warning("Zemberek API not available, attempting to start server...")
            self._start_api_server()
            
    def _init_subprocess_mode(self):
        """Initialize subprocess/CLI mode"""
        # Check if Zemberek CLI is available
        self.zemberek_cmd = self._find_zemberek_cli()
        if not self.zemberek_cmd:
            logger.warning("Zemberek CLI not found, falling back to REST API")
            self.mode = ZemberekMode.REST_API
            self._init_api_mode()
            
    def _find_zemberek_cli(self) -> Optional[str]:
        """Find Zemberek CLI executable"""
        possible_paths = [
            "zemberek",
            "./zemberek-cli",
            "/usr/local/bin/zemberek",
            str(Path.home() / "zemberek" / "bin" / "zemberek")
        ]
        
        for path in possible_paths:
            if os.path.exists(path) or subprocess.run(
                ["which", path], capture_output=True
            ).returncode == 0:
                return path
                
        return None
        
    def _start_api_server(self):
        """Start Zemberek API server"""
        try:
            # Try to start server in background
            subprocess.Popen([
                "java", "-jar", self.config.zemberek_jar_path,
                "server", "--port", "4567"
            ])
            
            # Wait for server to start
            import time
            time.sleep(3)
            
            # Test connection again
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Zemberek API server started successfully")
            else:
                raise Exception("Server started but not responding")
                
        except Exception as e:
            logger.error(f"Failed to start Zemberek API server: {e}")
            raise
            
    def analyze_morphology(self, text: str) -> List[MorphAnalysis]:
        """Perform morphological analysis on text"""
        # Check cache
        if self._cache and text in self._cache:
            return self._cache[text]
            
        result = []
        
        if self.mode == ZemberekMode.JAVA_DIRECT:
            result = self._analyze_morphology_java(text)
        elif self.mode == ZemberekMode.REST_API:
            result = self._analyze_morphology_api(text)
        elif self.mode == ZemberekMode.SUBPROCESS:
            result = self._analyze_morphology_subprocess(text)
            
        # Cache result
        if self._cache is not None and len(self._cache) < self.config.cache_size:
            self._cache[text] = result
            
        return result
        
    def _analyze_morphology_java(self, text: str) -> List[MorphAnalysis]:
        """Morphological analysis using Java integration"""
        results = []
        
        # Tokenize
        tokens = self.tokenizer.tokenizeToStrings(JString(text))
        
        for token in tokens:
            analyses = self.morphology.analyze(token)
            
            if analyses:
                # Use first analysis (most likely)
                analysis = analyses[0]
                
                result = MorphAnalysis(
                    surface=str(token),
                    lemma=str(analysis.getLemmas()[0]),
                    pos=str(analysis.getPos()),
                    morphemes=[str(m) for m in analysis.getMorphemes()],
                    formatted=str(analysis.formatLong()),
                    stem=str(analysis.getStems()[0]),
                    endings=[str(e) for e in analysis.getEndings()]
                )
                
                results.append(result)
                
        return results
        
    def _analyze_morphology_api(self, text: str) -> List[MorphAnalysis]:
        """Morphological analysis using REST API"""
        try:
            response = requests.post(
                f"{self.api_url}/morphology",
                json={"text": text},
                timeout=self.config.api_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_api_morphology(data)
            else:
                logger.error(f"API error: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []
            
    def _analyze_morphology_subprocess(self, text: str) -> List[MorphAnalysis]:
        """Morphological analysis using subprocess"""
        try:
            result = subprocess.run(
                [self.zemberek_cmd, "morphology", "-i", text],
                capture_output=True,
                text=True,
                timeout=self.config.api_timeout
            )
            
            if result.returncode == 0:
                return self._parse_cli_morphology(result.stdout)
            else:
                logger.error(f"CLI error: {result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            logger.error("CLI command timed out")
            return []
            
    def _parse_api_morphology(self, data: Dict) -> List[MorphAnalysis]:
        """Parse API morphology response"""
        results = []
        
        for item in data.get("results", []):
            analysis = MorphAnalysis(
                surface=item.get("surface", ""),
                lemma=item.get("lemma", ""),
                pos=item.get("pos", ""),
                morphemes=item.get("morphemes", []),
                formatted=item.get("formatted", ""),
                stem=item.get("stem", ""),
                endings=item.get("endings", []),
                attributes=item.get("attributes", {})
            )
            results.append(analysis)
            
        return results
        
    def _parse_cli_morphology(self, output: str) -> List[MorphAnalysis]:
        """Parse CLI morphology output"""
        results = []
        lines = output.strip().split("\n")
        
        for line in lines:
            if line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    analysis = MorphAnalysis(
                        surface=parts[0],
                        lemma=parts[1] if len(parts) > 1 else "",
                        pos=parts[2] if len(parts) > 2 else "",
                        morphemes=parts[3].split("+") if len(parts) > 3 else [],
                        formatted=line,
                        stem=parts[1] if len(parts) > 1 else "",
                        endings=[]
                    )
                    results.append(analysis)
                    
        return results
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Turkish text"""
        if self.mode == ZemberekMode.JAVA_DIRECT:
            return list(self.tokenizer.tokenizeToStrings(JString(text)))
        elif self.mode == ZemberekMode.REST_API:
            response = requests.post(
                f"{self.api_url}/tokenize",
                json={"text": text},
                timeout=self.config.api_timeout
            )
            if response.status_code == 200:
                return response.json().get("tokens", [])
        elif self.mode == ZemberekMode.SUBPROCESS:
            result = subprocess.run(
                [self.zemberek_cmd, "tokenize", "-i", text],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip().split()
                
        # Fallback to simple tokenization
        return text.split()
        
    def normalize(self, text: str) -> str:
        """Normalize Turkish text (deasciify, correct typos)"""
        if self.mode == ZemberekMode.REST_API:
            response = requests.post(
                f"{self.api_url}/normalize",
                json={"text": text},
                timeout=self.config.api_timeout
            )
            if response.status_code == 200:
                return response.json().get("normalized", text)
                
        # Fallback: return original
        return text
        
    def spell_check(self, text: str) -> List[Dict[str, Any]]:
        """Check spelling and suggest corrections"""
        suggestions = []
        
        if self.mode == ZemberekMode.JAVA_DIRECT and hasattr(self, 'spell_checker'):
            tokens = self.tokenize(text)
            for token in tokens:
                if not self.spell_checker.check(JString(token)):
                    suggestions.append({
                        "word": token,
                        "suggestions": list(self.spell_checker.suggestForWord(JString(token)))
                    })
                    
        elif self.mode == ZemberekMode.REST_API:
            response = requests.post(
                f"{self.api_url}/spell_check",
                json={"text": text},
                timeout=self.config.api_timeout
            )
            if response.status_code == 200:
                suggestions = response.json().get("corrections", [])
                
        return suggestions
        
    def disambiguate(self, text: str) -> List[str]:
        """Morphological disambiguation"""
        if self.mode == ZemberekMode.REST_API:
            response = requests.post(
                f"{self.api_url}/disambiguate",
                json={"text": text},
                timeout=self.config.api_timeout
            )
            if response.status_code == 200:
                return response.json().get("disambiguated", [])
                
        # Fallback: return morphology analysis
        analyses = self.analyze_morphology(text)
        return [a.formatted for a in analyses]
        
    def find_named_entities(self, text: str) -> List[Dict[str, str]]:
        """Named entity recognition"""
        entities = []
        
        if self.mode == ZemberekMode.REST_API:
            response = requests.post(
                f"{self.api_url}/ner",
                json={"text": text},
                timeout=self.config.api_timeout
            )
            if response.status_code == 200:
                entities = response.json().get("entities", [])
                
        return entities
        
    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        """Complete sentence analysis"""
        tokens = self.tokenize(sentence)
        morphology = self.analyze_morphology(sentence)
        disambiguation = self.disambiguate(sentence)
        
        return SentenceAnalysis(
            text=sentence,
            tokens=tokens,
            morphology=morphology,
            disambiguation=disambiguation
        )
        
    def close(self):
        """Clean up resources"""
        if self.mode == ZemberekMode.JAVA_DIRECT and jpype.isJVMStarted():
            jpype.shutdownJVM()


class ZemberekServer:
    """Standalone Zemberek REST API server"""
    
    def __init__(self, port: int = 4567):
        self.port = port
        self.app = None
        self._init_flask_app()
        
    def _init_flask_app(self):
        """Initialize Flask application"""
        try:
            from flask import Flask, request, jsonify
            
            self.app = Flask(__name__)
            self.zemberek = ZemberekIntegration(
                ZemberekConfig(mode=ZemberekMode.JAVA_DIRECT)
            )
            
            @self.app.route('/health', methods=['GET'])
            def health():
                return jsonify({"status": "healthy"})
                
            @self.app.route('/morphology', methods=['POST'])
            def morphology():
                data = request.json
                text = data.get('text', '')
                analyses = self.zemberek.analyze_morphology(text)
                
                results = []
                for a in analyses:
                    results.append({
                        "surface": a.surface,
                        "lemma": a.lemma,
                        "pos": a.pos,
                        "morphemes": a.morphemes,
                        "formatted": a.formatted,
                        "stem": a.stem,
                        "endings": a.endings
                    })
                    
                return jsonify({"results": results})
                
            @self.app.route('/tokenize', methods=['POST'])
            def tokenize():
                data = request.json
                text = data.get('text', '')
                tokens = self.zemberek.tokenize(text)
                return jsonify({"tokens": tokens})
                
            @self.app.route('/normalize', methods=['POST'])
            def normalize():
                data = request.json
                text = data.get('text', '')
                normalized = self.zemberek.normalize(text)
                return jsonify({"normalized": normalized})
                
            @self.app.route('/spell_check', methods=['POST'])
            def spell_check():
                data = request.json
                text = data.get('text', '')
                corrections = self.zemberek.spell_check(text)
                return jsonify({"corrections": corrections})
                
            @self.app.route('/disambiguate', methods=['POST'])
            def disambiguate():
                data = request.json
                text = data.get('text', '')
                disambiguated = self.zemberek.disambiguate(text)
                return jsonify({"disambiguated": disambiguated})
                
            @self.app.route('/ner', methods=['POST'])
            def ner():
                data = request.json
                text = data.get('text', '')
                entities = self.zemberek.find_named_entities(text)
                return jsonify({"entities": entities})
                
        except ImportError:
            logger.error("Flask not available for REST API server")
            
    def run(self):
        """Run the server"""
        if self.app:
            self.app.run(host='0.0.0.0', port=self.port)
        else:
            logger.error("Flask app not initialized")


def test_zemberek_integration():
    """Test Zemberek integration"""
    print("Testing Zemberek Integration")
    print("=" * 50)
    
    # Initialize Zemberek
    zemberek = ZemberekIntegration()
    
    test_sentences = [
        "Türkiye'nin başkenti Ankara'dır.",
        "Öğrenciler sınavlara hazırlanıyorlar.",
        "Kitapları masanın üzerine koydum.",
        "Yarın hava güzel olacakmış.",
        "İstanbul'dan Ankara'ya gidiyorum."
    ]
    
    for sentence in test_sentences:
        print(f"\nAnalyzing: {sentence}")
        print("-" * 40)
        
        # Tokenization
        tokens = zemberek.tokenize(sentence)
        print(f"Tokens: {tokens}")
        
        # Morphological analysis
        analyses = zemberek.analyze_morphology(sentence)
        for analysis in analyses:
            print(f"  {analysis.surface}:")
            print(f"    Lemma: {analysis.lemma}")
            print(f"    POS: {analysis.pos}")
            print(f"    Morphemes: {'+'.join(analysis.morphemes)}")
            
        # Spell check
        corrections = zemberek.spell_check(sentence)
        if corrections:
            print(f"Spelling corrections: {corrections}")
            
    zemberek.close()


if __name__ == "__main__":
    # Test integration
    test_zemberek_integration()
    
    # Optionally start server
    # server = ZemberekServer()
    # server.run()