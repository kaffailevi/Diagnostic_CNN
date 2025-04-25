import {Component, OnInit} from '@angular/core';
import {Button} from 'primeng/button';
import {BASE_URL} from '../../util/urls';
import {Image} from '../../models/image';
import {ApiService} from '../../service/api.service';
import {MyImage} from '../../models/my-image';
import {FileSelectEvent, FileUpload, FileUploadEvent} from 'primeng/fileupload';
import {Badge} from 'primeng/badge';
import {ProgressBar} from 'primeng/progressbar';
import {DecimalPipe, NgForOf, NgIf} from '@angular/common';

@Component({
  selector: 'app-profile',
  imports: [
    Button,
    FileUpload,
    Badge,
    NgForOf,
    NgIf,
    DecimalPipe
  ],
  templateUrl: './profile.component.html',
  styleUrl: './profile.component.scss'
})
export class ProfileComponent implements OnInit {


  protected readonly BASE_URL = BASE_URL;
  images: MyImage[] =[];
  totalSizePercent: any;
  totalSize: number = 0;
  uploadedFiles: File[] = [];
  constructor(private apiService: ApiService) {
  }

  ngOnInit(): void {
    this.getMyImages();
  }


  predict($img: MyImage) {

  }

  private getMyImages() {
    this.apiService.getImages().subscribe((images: MyImage[]) => {
      this.images = images;
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
}
