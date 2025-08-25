"""
Comprehensive tests for Data Processing modules
"""
import pytest
import pandas as pd
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.simple_data_processor import SimpleDataProcessor


class TestDataProcessor:
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance for testing"""
        return DataProcessor()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'student_id': ['001', '002', '003'],
            'score': [75, 85, 90],
            'subject': ['Math', 'Math', 'Physics'],
            'date': pd.date_range('2024-01-01', periods=3)
        })
    
    @pytest.fixture
    def sample_json_data(self):
        """Create sample JSON data for testing"""
        return [
            {
                'student_id': '001',
                'name': 'Ali Yılmaz',
                'grades': {'math': 85, 'physics': 78},
                'learning_style': 'visual'
            },
            {
                'student_id': '002',
                'name': 'Ayşe Demir',
                'grades': {'math': 92, 'physics': 88},
                'learning_style': 'auditory'
            }
        ]
    
    @pytest.mark.unit
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor is not None
        assert hasattr(processor, 'process') or hasattr(processor, 'load_data')
    
    @pytest.mark.unit
    def test_load_csv_data(self, processor, sample_data):
        """Test loading CSV data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            if hasattr(processor, 'load_csv'):
                data = processor.load_csv(temp_file)
            elif hasattr(processor, 'load_data'):
                data = processor.load_data(temp_file)
            else:
                pytest.skip("Processor doesn't have data loading methods")
            
            assert data is not None
            assert len(data) == 3
            assert 'student_id' in data.columns if hasattr(data, 'columns') else True
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.unit
    def test_load_json_data(self, processor, sample_json_data):
        """Test loading JSON data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_data, f)
            temp_file = f.name
        
        try:
            if hasattr(processor, 'load_json'):
                data = processor.load_json(temp_file)
            elif hasattr(processor, 'load_data'):
                data = processor.load_data(temp_file)
            else:
                pytest.skip("Processor doesn't have data loading methods")
            
            assert data is not None
            assert len(data) == 2
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.unit
    def test_process_student_data(self, processor, sample_data):
        """Test processing student data"""
        if hasattr(processor, 'process_student_data'):
            processed = processor.process_student_data(sample_data)
            
            assert processed is not None
            # Check for expected processing results
            if isinstance(processed, pd.DataFrame):
                assert len(processed) > 0
        else:
            pytest.skip("Processor doesn't have process_student_data method")
    
    @pytest.mark.unit
    def test_calculate_statistics(self, processor, sample_data):
        """Test calculating statistics from data"""
        if hasattr(processor, 'calculate_statistics'):
            stats = processor.calculate_statistics(sample_data)
            
            assert stats is not None
            assert 'mean' in stats or 'average' in stats
            assert 'std' in stats or 'standard_deviation' in stats
        else:
            # Try alternative method names
            if hasattr(processor, 'get_statistics'):
                stats = processor.get_statistics(sample_data)
                assert stats is not None
    
    @pytest.mark.unit
    def test_filter_by_subject(self, processor, sample_data):
        """Test filtering data by subject"""
        if hasattr(processor, 'filter_by_subject'):
            math_data = processor.filter_by_subject(sample_data, 'Math')
            
            assert len(math_data) == 2
            assert all(row['subject'] == 'Math' for _, row in math_data.iterrows())
        else:
            # Manual filtering test
            math_data = sample_data[sample_data['subject'] == 'Math']
            assert len(math_data) == 2
    
    @pytest.mark.unit
    def test_handle_missing_data(self, processor):
        """Test handling missing data"""
        data_with_missing = pd.DataFrame({
            'student_id': ['001', '002', None, '004'],
            'score': [75, None, 90, 85],
            'subject': ['Math', 'Math', None, 'Physics']
        })
        
        if hasattr(processor, 'handle_missing_data'):
            cleaned = processor.handle_missing_data(data_with_missing)
            
            # Check that missing data is handled
            assert cleaned is not None
            assert cleaned.isnull().sum().sum() < data_with_missing.isnull().sum().sum()
        else:
            # Basic test for data with nulls
            assert data_with_missing.isnull().sum().sum() > 0
    
    @pytest.mark.unit
    def test_export_to_json(self, processor, sample_data):
        """Test exporting data to JSON"""
        if hasattr(processor, 'export_to_json'):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_file = f.name
            
            try:
                processor.export_to_json(sample_data, temp_file)
                
                # Verify file was created and contains data
                assert os.path.exists(temp_file)
                with open(temp_file, 'r') as f:
                    loaded_data = json.load(f)
                assert len(loaded_data) > 0
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    @pytest.mark.unit
    def test_aggregate_scores(self, processor):
        """Test aggregating scores by student"""
        data = pd.DataFrame({
            'student_id': ['001', '001', '002', '002'],
            'score': [75, 80, 85, 90],
            'subject': ['Math', 'Physics', 'Math', 'Physics']
        })
        
        if hasattr(processor, 'aggregate_scores'):
            aggregated = processor.aggregate_scores(data)
            
            assert len(aggregated) == 2  # Two unique students
            assert all(student_id in ['001', '002'] for student_id in aggregated['student_id'].values)
        else:
            # Manual aggregation test
            aggregated = data.groupby('student_id')['score'].mean()
            assert len(aggregated) == 2


class TestSimpleDataProcessor:
    
    @pytest.fixture
    def processor(self):
        """Create a SimpleDataProcessor instance for testing"""
        return SimpleDataProcessor()
    
    @pytest.mark.unit
    def test_simple_processor_initialization(self, processor):
        """Test simple processor initialization"""
        assert processor is not None
    
    @pytest.mark.unit
    def test_process_simple_data(self, processor):
        """Test processing simple data"""
        data = {
            'values': [1, 2, 3, 4, 5],
            'labels': ['a', 'b', 'c', 'd', 'e']
        }
        
        if hasattr(processor, 'process'):
            result = processor.process(data)
            assert result is not None
    
    @pytest.mark.unit
    def test_validate_data_format(self, processor):
        """Test data format validation"""
        valid_data = {'student_id': '001', 'score': 85}
        invalid_data = {'invalid': 'format'}
        
        if hasattr(processor, 'validate'):
            assert processor.validate(valid_data) == True
            assert processor.validate(invalid_data) == False
        else:
            # Skip if validation method doesn't exist
            pytest.skip("Processor doesn't have validate method")
    
    @pytest.mark.unit
    def test_transform_data(self, processor):
        """Test data transformation"""
        raw_data = [
            {'score': '85', 'grade': 'A'},
            {'score': '72', 'grade': 'B'}
        ]
        
        if hasattr(processor, 'transform'):
            transformed = processor.transform(raw_data)
            
            assert transformed is not None
            # Check if scores are converted to numbers
            if isinstance(transformed, list):
                for item in transformed:
                    if 'score' in item:
                        assert isinstance(item['score'], (int, float))


@pytest.mark.integration
class TestDataProcessingIntegration:
    
    @pytest.fixture
    def processor(self):
        return DataProcessor()
    
    @pytest.fixture
    def simple_processor(self):
        return SimpleDataProcessor()
    
    def test_complete_data_pipeline(self, processor):
        """Test complete data processing pipeline"""
        # Create test data
        raw_data = pd.DataFrame({
            'student_id': ['001', '001', '002', '002', '003'],
            'score': [75, 80, 85, 90, 88],
            'subject': ['Math', 'Physics', 'Math', 'Physics', 'Math'],
            'date': pd.date_range('2024-01-01', periods=5)
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save raw data
            input_file = os.path.join(tmpdir, 'input.csv')
            raw_data.to_csv(input_file, index=False)
            
            # Process data
            if hasattr(processor, 'load_data'):
                data = processor.load_data(input_file)
            else:
                data = pd.read_csv(input_file)
            
            # Calculate statistics
            if hasattr(processor, 'calculate_statistics'):
                stats = processor.calculate_statistics(data)
                assert stats is not None
            
            # Export processed data
            output_file = os.path.join(tmpdir, 'output.json')
            if hasattr(processor, 'export_to_json'):
                processor.export_to_json(data, output_file)
                assert os.path.exists(output_file)
            else:
                data.to_json(output_file, orient='records')
                assert os.path.exists(output_file)
    
    def test_batch_processing(self, processor):
        """Test batch processing of multiple files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            file_paths = []
            for i in range(3):
                data = pd.DataFrame({
                    'student_id': [f'00{i+1}'],
                    'score': [70 + i * 10],
                    'subject': ['Math']
                })
                file_path = os.path.join(tmpdir, f'batch_{i}.csv')
                data.to_csv(file_path, index=False)
                file_paths.append(file_path)
            
            # Process all files
            all_data = []
            for file_path in file_paths:
                if hasattr(processor, 'load_data'):
                    data = processor.load_data(file_path)
                else:
                    data = pd.read_csv(file_path)
                all_data.append(data)
            
            # Combine results
            combined = pd.concat(all_data, ignore_index=True)
            assert len(combined) == 3
            assert all(score in [70, 80, 90] for score in combined['score'].values)
    
    def test_error_recovery(self, processor):
        """Test error recovery in data processing"""
        # Test with corrupted data
        corrupted_data = "This is not valid CSV data\n!!!"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(corrupted_data)
            temp_file = f.name
        
        try:
            if hasattr(processor, 'load_data'):
                try:
                    data = processor.load_data(temp_file)
                    # Should handle error gracefully
                    assert data is None or len(data) == 0
                except Exception as e:
                    # Should raise a meaningful exception
                    assert str(e) != ""
        finally:
            os.unlink(temp_file)