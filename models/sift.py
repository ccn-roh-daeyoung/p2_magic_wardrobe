import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

class SIFTReranker:
    def __init__(self, match_threshold: float = 0.7, min_matches: int = 10):
        """
        SIFT 기반 이미지 랭크 재정렬 클래스 초기화
        
        Args:
            match_threshold (float): 특징점 매칭을 위한 거리 임계값 (낮을수록 엄격한 매칭)
            min_matches (int): 유의미한 매칭으로 간주되는 최소 매칭 수
        """
        self.sift = cv2.SIFT_create()
        self.match_threshold = match_threshold
        self.min_matches = min_matches
    
    def compute_sift_features(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        주어진 이미지의 SIFT 특징점과 디스크립터 계산
        
        Args:
            image_path (str): 이미지 파일 경로
        
        Returns:
            tuple: (키포인트, 디스크립터)
        """
        # 이미지 읽기 및 그레이스케일 변환
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
        
        # SIFT 특징점 및 디스크립터 계산
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> int:
        """
        두 이미지의 SIFT 디스크립터 간 매칭 수 계산
        
        Args:
            desc1 (np.ndarray): 첫 번째 이미지의 디스크립터
            desc2 (np.ndarray): 두 번째 이미지의 디스크립터
        
        Returns:
            int: 매칭된 특징점의 수
        """
        # FLANN 기반 매처 생성
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 매칭 수행 (K-NN 매칭)
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test 적용
        good_matches = 0
        for m, n in matches:
            if m.distance < self.match_threshold * n.distance:
                good_matches += 1
        
        return good_matches
    
    def rerank_images(self, input_image_path: str, category_urls: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        SIFT 기반으로 각 카테고리의 이미지 랭크 재정렬
        
        Args:
            input_image_path (str): 입력 이미지 경로
            category_urls (dict): 카테고리별 이미지 URL 및 정보
        
        Returns:
            dict: SIFT 매칭 기반으로 재정렬된 카테고리별 이미지 URL 및 정보
        """
        # 입력 이미지의 SIFT 특징점 계산
        try:
            input_keypoints, input_descriptors = self.compute_sift_features(input_image_path)
        except Exception as e:
            print(f"입력 이미지 SIFT 특징점 계산 중 오류: {e}")
            return category_urls
        
        # 재정렬된 결과를 저장할 딕셔너리
        reranked_category_urls = {}
        
        # 각 카테고리별로 처리
        for category, urls in category_urls.items():
            # 현재 카테고리의 이미지들에 대해 SIFT 매칭 수행
            sift_matched_urls = []
            
            for item in urls:
                try:
                    # 이미지 URL로부터 로컬 파일 경로 추론 (실제 구현 시 적절히 수정 필요)
                    image_local_path = self._url_to_local_path(item['url'])
                    
                    # 비교 대상 이미지의 SIFT 특징점 계산
                    ref_keypoints, ref_descriptors = self.compute_sift_features(image_local_path)
                    
                    # 특징점 매칭 수 계산
                    match_count = self.match_features(input_descriptors, ref_descriptors)
                    
                    # 매칭 수가 임계값 이상인 경우에만 추가
                    if match_count >= self.min_matches:
                        sift_matched_urls.append({
                            **item,
                            'sift_match_count': match_count
                        })
                except Exception as e:
                    print(f"이미지 SIFT 처리 중 오류: {e}")
            
            # SIFT 매칭 수를 기준으로 정렬 (매칭 수 내림차순)
            sift_matched_urls.sort(key=lambda x: x.get('sift_match_count', 0), reverse=True)
            
            # 원래 확률과 SIFT 매칭 수를 결합하여 최종 랭킹 결정 (가중 평균 등의 방식 사용 가능)
            weighted_matched_urls = []
            for item in sift_matched_urls:
                sift_weight = 0.5  # SIFT 매칭 수의 가중치 (필요에 따라 조정)
                prob_weight = 0.5  # 원래 확률의 가중치
                
                weighted_score = (
                    sift_weight * (item.get('sift_match_count', 0) / max(1, len(sift_matched_urls))) + 
                    prob_weight * item.get('probability', 0)
                )
                
                weighted_matched_urls.append({
                    **item,
                    'weighted_score': weighted_score
                })
            
            # 가중치 점수 기준으로 최종 정렬
            weighted_matched_urls.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
            
            # 최종 결과에서 불필요한 키 제거
            reranked_category_urls[category] = [
                {k: v for k, v in item.items() if k in ['url', 'class_name', 'probability']} 
                for item in weighted_matched_urls
            ]
        
        return reranked_category_urls
    
    def _url_to_local_path(self, url: str) -> str:
        """
        이미지 URL을 로컬 파일 경로로 변환 (실제 구현 시 적절히 수정 필요)
        
        Args:
            url (str): 이미지 URL
        
        Returns:
            str: 로컬 이미지 파일 경로
        """
        # 실제 구현에서는 URL을 로컬 파일 경로로 변환하는 로직 필요
        # 예: 다운로드, 캐싱, 경로 매핑 등
        # 여기서는 간단한 예시로 URL 자체를 파일 경로로 간주 (실제 사용 시 수정 필요)
        return url

# 사용 예시
def rerank_fashion_results(input_image_path: str, category_urls: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    SIFT 기반 이미지 랭크 재정렬을 위한 래퍼 함수
    
    Args:
        input_image_path (str): 입력 이미지 경로
        category_urls (dict): 카테고리별 이미지 URL 및 정보
    
    Returns:
        dict: SIFT 매칭 기반으로 재정렬된 카테고리별 이미지 URL 및 정보
    """
    reranker = SIFTReranker()
    return reranker.rerank_images(input_image_path, category_urls)