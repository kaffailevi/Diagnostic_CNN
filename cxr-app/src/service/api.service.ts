import {HttpClient} from '@angular/common/http';
import {Injectable} from '@angular/core';
import {Observable} from 'rxjs';
import {BASE_URL} from '../util/urls';
import {Image} from '../models/image';
import {ImageCount} from '../models/image-count';
import {Prediction} from '../models/prediction';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  constructor(private http: HttpClient) {
  }


  getImages(): Observable<string[]> {
    return this.http.get<string[]>(`${BASE_URL}/my-images`, {withCredentials: true});
  }

  getImage(id: string): Observable<string> {
    return this.http.get<string>(`${BASE_URL}/my-images/${id}`, {withCredentials: true});
  }

  // http://localhost:8000/images/?skip=0&limit=10&type=images&category=COVID example for url, category and type are optional
  // skip is the number of images to skip, limit is the number of images to return
  // type is the type of images to return, category is the category of images to return
  // type can be 'images' or 'masks', category can be 'COVID' or 'Normal' , 'Viral Pneumonia' or 'Lung_Opacity'

  getImagesWithPagination(skip: number, limit: number, type?: string, category?: string): Observable<Image[]> {
    let params = new URLSearchParams();
    params.append('skip', skip.toString());
    params.append('limit', limit.toString());
    if (type) params.append('type', type);
    if (category) params.append('category', category);

    return this.http.get<Image[]>(`${BASE_URL}/images/?${params.toString()}`, {withCredentials: true});
  }

  // http://localhost:8000/image_count/?category=normal&type=images

  getImageCount(type?: string, category?: string): Observable<ImageCount> {
    let params = new URLSearchParams();
    if (type) params.append('type', type);
    if (category) params.append('category', category);
    return this.http.get<ImageCount>(`${BASE_URL}/image_count/?${params.toString()}`, {withCredentials: true});
  }

  getImagePrediction(image: Image, model_name: string): Observable<Prediction> {
    let params = new URLSearchParams();
    params.append('image_path', image.filename);
    params.append('model_name', model_name);
    return this.http.post<Prediction>(`${BASE_URL}/predict_stored/?${params.toString()}`, {withCredentials: true});
  }

  // http://localhost:8000/masked_segment/?filename=Lung_Opacity%2Fimages%2F20812.png&mode=extract
  getPredictedMaskUrl(image: Image, mode?: string): string {
    let params = new URLSearchParams();
    params.append('filename', image.filename);
    if (mode) params.append('mode', mode);
    else
      params.append('mode', 'extract');
    return `${BASE_URL}/masked_segment/?${params.toString()}`;
  }

}
