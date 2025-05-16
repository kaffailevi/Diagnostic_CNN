import {Component, OnInit} from '@angular/core';
import {Button} from 'primeng/button';
import {BASE_URL} from '../../util/urls';
import {ApiService} from '../../service/api.service';
import {MyImage} from '../../models/my-image';
import {FileSelectEvent, FileUpload} from 'primeng/fileupload';
import {Badge} from 'primeng/badge';
import {DecimalPipe, JsonPipe, NgForOf, NgIf} from '@angular/common';
import {DropdownModule} from 'primeng/dropdown';
import {FormsModule} from '@angular/forms';
import {Dialog} from 'primeng/dialog';
import {Prediction} from '../../models/prediction';
import {AutoComplete, AutoCompleteCompleteEvent, AutoCompleteSelectEvent} from 'primeng/autocomplete';

@Component({
  selector: 'app-profile',
  imports: [
    Button,
    FileUpload,
    Badge,
    NgForOf,
    NgIf,
    DecimalPipe,
    DropdownModule,
    FormsModule,
    Dialog,
    JsonPipe,
    AutoComplete
  ],
  templateUrl: './profile.component.html',
  styleUrl: './profile.component.scss'
})
export class ProfileComponent implements OnInit {


  protected readonly BASE_URL = BASE_URL;
  images: MyImage[] = [];
  totalSizePercent: any;
  totalSize: number = 0;
  uploadedFiles: File[] = [];
  selectedModel: any = "";
  model_names: string[] = [];
  mask_url: string = "";
  visible: boolean = false;
  json: Prediction = {
    prediction: '',
    confidence_scores: {},
    image_path: ''
  };
  selectedImage!: MyImage;
  filteredImages: MyImage[] = [];
  searchTerm: string = '';

  constructor(private apiService: ApiService) {
  }

  ngOnInit(): void {
    this.getMyImages();
    this.getModelNames();
  }


  predict($img: MyImage) {
    this.visible = true;

    this.apiService.getImagePredictionFromDb($img, this.selectedModel).subscribe((response: any) => {
      console.log(response);
      this.json = response;
      this.mask_url = `${BASE_URL}/masked_segment_db/${$img.id}/?mode=overlay`;
    });
  }

  private getMyImages() {
    this.apiService.getMyImages().subscribe((images: MyImage[]) => {
      this.images = images;
      this.filteredImages = images;
    });
  }


  onRemoveTemplatingFile($event: MouseEvent, file: File, removeFileCallback: any, index: any) {
    $event.preventDefault();
    removeFileCallback(index);
    this.uploadedFiles.splice(index, 1);
    this.totalSize -= file.size;
    this.totalSizePercent = (this.totalSize / 1000000) * 100;
  }

  onTemplatedUpload() {
    this.getMyImages();
    this.totalSize = 0;
    this.totalSizePercent = 0;
    this.uploadedFiles = [];
  }

  onSelectedFiles($event: FileSelectEvent) {
    this.totalSize = 0;
    this.uploadedFiles = [...$event.files];
    for (let file of $event.files) {
      this.totalSize += file.size;
    }
    this.totalSizePercent = (this.totalSize / 1000000) * 100;
  }

  choose($event: MouseEvent, chooseCallback: any) {
    $event.preventDefault();
    chooseCallback();
  }

  uploadEvent(uploadCallback: any) {
    uploadCallback();
    this.totalSize = 0;
    this.totalSizePercent = 0;
    // this.apiService.uploadImages(this.uploadedFiles).subscribe((response: any) => {
    //   console.log(response);
    //   window.location.reload();
    // });
  }

  delete($img: MyImage) {
    this.apiService.deleteImage($img).subscribe((response: any) => {
      console.log(response);
      this.getMyImages();
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

  searchImages(event: AutoCompleteCompleteEvent) {
    const q = (event.query || '').trim().toLowerCase();

    if ( q.length === 0 || q === '') {
      this.clearSearch();
      this.filteredImages = [...this.images];
      return;
    }

    this.filteredImages = this.images.filter(img =>
      img.filename.toLowerCase().includes(q)
    );
  }


  get displayedImages(): MyImage[] {
    if (this.selectedImage) {
      return [this.selectedImage];
    }
    return this.filteredImages;
  }

  onImageSelect(event: AutoCompleteSelectEvent) {
    const image: MyImage = event.value;
    this.selectedImage = image;
    this.searchTerm = image.filename;
  }


  clearSearch() {
    this.searchTerm = '';
    this.selectedImage = undefined!;
    this.filteredImages = [...this.images];
  }


  onKeyUp($event: KeyboardEvent) {
    if(this.searchTerm === '' || $event.key === 'Escape') {
      this.clearSearch();
    }

  }
}
