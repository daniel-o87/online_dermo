import requests
import json
import os

def test_api():
    # Test health check
    try:
        response = requests.get("http://localhost:8008/ping")
        print("Health check response:", response.json())
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return

    # Test prediction
    try:
        # Verify file exists
        dicom_file = 'ISIC_8580508.dcm'
        if not os.path.exists(dicom_file):
            print(f"Error: DICOM file '{dicom_file}' not found")
            return

        # Prepare metadata
        metadata = {
            'sex': 'male',
            'anatom_site_general_challenge': 'torso',
            'age_approx': 70.0
        }

        # Open and prepare files
        with open(dicom_file, 'rb') as f:
            files = {
                'file': (dicom_file, f, 'application/dicom'),
                'metadata': (None, json.dumps(metadata))
            }
            
            # Make prediction request
            response = requests.post("http://localhost:8008/predict", files=files)
            
            # Check response
            if response.status_code == 200:
                print("\nPrediction response:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"\nError response (status code {response.status_code}):")
                print(json.dumps(response.json(), indent=2))
                if 'detail' in response.json():
                    print("\nError detail:", response.json()['detail'])

    except Exception as e:
        print(f"Prediction request failed: {str(e)}")

if __name__ == "__main__":
    test_api()
