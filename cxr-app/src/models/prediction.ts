export interface Prediction {
  prediction: string;
  confidence_scores: {
    [key: string]: number;
  };
  image_path: string;
}
