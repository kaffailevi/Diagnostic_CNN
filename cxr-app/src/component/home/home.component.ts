import {Component, OnInit} from '@angular/core';
import {ButtonModule} from 'primeng/button';
import {PaginatorModule, PaginatorState} from 'primeng/paginator';
import {AuthService} from '../../service/auth.service';
import {ApiService} from '../../service/api.service';
import {JsonPipe} from '@angular/common';
import {Image} from '../../models/image';
import {BASE_URL} from '../../util/urls';
import {ImageCount} from '../../models/image-count';
import {DropdownModule} from 'primeng/dropdown';
import {FormsModule} from '@angular/forms';
import {Prediction} from '../../models/prediction';
import {Dialog} from 'primeng/dialog';


@Component({
  selector: 'app-home',
  imports: [ButtonModule, PaginatorModule, DropdownModule, FormsModule, Dialog, JsonPipe],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent implements OnInit {
  first: number = 0;
  rows = 10;
  isAuthenticated: Function;
  images: Image[] = [];
  imageCount: ImageCount = {
    count: 0,
    type: '',
    category: ''
  };


  categories = [{label: 'Covid', value: 'COVID'},
    {label: 'Normal', value: 'Normal'},
    {label: 'Viral Pneumonia', value: 'Viral Pneumonia'},
    {label: 'Lung Opacity', value: 'Lung_Opacity'},
    {label: 'All', value: ''}
  ];
  selectedCategory = {label: 'All', value: ''};
  protected readonly BASE_URL = BASE_URL;
  visible: boolean = false;
  json: Prediction = {
    prediction: '',
    confidence_scores: {},
    image_path: ''
  }
  mask_url: string = '';
  model_names: string[] | undefined;
  selectedModel: string | undefined;


  constructor(private authService: AuthService, private apiService: ApiService) {
    this.isAuthenticated = this.authService.isAuthenticated;
  }


  ngOnInit() {
    this.getImageCount();
    this.getPublicImages()
    this.getModelNames();
  }

  onPageChange($event: PaginatorState) {
    this.first = $event.first ?? 1;
    this.rows = $event.rows ?? 25;
    this.getPublicImages()
  }

  login() {
    this.authService.login();
  }

  getPublicImages() {
    let category: string | undefined;
    let type = 'images';
    if (this.selectedCategory.value !== '')
      category = this.selectedCategory.value;
    this.apiService.getImagesWithPagination(this.first, this.rows, type, category).subscribe(images => {
      this.images = images;
    })
  }

  private getImageCount() {
    let category: string | undefined;

    if (this.selectedCategory.value !== '')
      category = this.selectedCategory.value;
    this.apiService.getImageCount(undefined, category).subscribe(imageCount => {
      this.imageCount = imageCount;
    })
  }

  onCategoryChange() {
    // Reset the paginator if the category is changed

    this.first = 0;
    this.apiService.getImagesWithPagination(this.first, this.rows, undefined, this.selectedCategory.value).subscribe(images => {
      this.images = images;
    })
    this.apiService.getImageCount(undefined, this.selectedCategory.value).subscribe(imageCount => {
      this.imageCount = imageCount;
    })

  }

  predict(image: Image) {

    this.mask_url = this.apiService.getPredictedMaskUrl(image, 'overlay');
    this.apiService.getImagePrediction(image, this.selectedModel ).subscribe((response: Prediction) => {
      console.log(response);
      this.json = response;
      // alert(`Prediction: ${response.prediction}`);
      this.visible = true;
    });
  }

  onModelChange() {

  }

  private getModelNames() {
    this.apiService.getModelNames().subscribe(model_names => {
      console.log(model_names);
      this.model_names = model_names;
    });
  }
}
