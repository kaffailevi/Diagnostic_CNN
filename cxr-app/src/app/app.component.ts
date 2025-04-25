import {Component, OnInit} from '@angular/core';
import { RouterOutlet } from '@angular/router';
import {Menubar} from 'primeng/menubar';
import {AuthService} from '../service/auth.service';
import {AsyncPipe, NgIf} from '@angular/common';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, Menubar, NgIf, AsyncPipe],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent  {
  title = 'cxr-app';
  menuItems = [
    { label: 'Dashboard', icon: 'pi pi-fw pi-home', routerLink: '/home' },
    { label: 'Profil', icon: 'pi pi-fw pi-user', routerLink: '/profile' },]
  isAuthenticated: Promise<boolean>;

  constructor(private authService: AuthService) {
    this.isAuthenticated = this.authService.isAuthenticated();
  }


}
