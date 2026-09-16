from insightface.app import FaceAnalysis


def load_face_model() -> FaceAnalysis:
	"""Create and prepare an InsightFace model for face analysis."""
	app = FaceAnalysis(
		name="buffalo_l",
		providers=["CPUExecutionProvider"],
		)
	app.prepare(
		ctx_id=0, 
			 det_size=(640, 640)
			 )
	return app


if __name__ == "__main__":
	load_face_model()
	print("InsightFace model loaded successfully")
   

