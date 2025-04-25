import {Component, OnInit} from '@angular/core';
import { RouterOutlet } from '@angular/router';
import {Menubar} from 'primeng/menubar';
import {AuthService} from '../service/auth.service';
import {AsyncPipe, NgIf} from '@angular/common';
import {ButtonDirective, ButtonLabel} from 'primeng/button';
import {PrimeTemplate} from 'primeng/api';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, Menubar, NgIf, AsyncPipe, ButtonDirective, ButtonLabel, PrimeTemplate],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent  {
  title = 'cxr-app';
  menuItems = [
    { label: 'Dashboard', icon: 'pi pi-fw pi-home', routerLink: '/home' },
    { label: 'Profil', icon: 'pi pi-fw pi-user', routerLink: '/profile' },]
  isAuthenticated: Promise<boolean>;
  publicMenuItems = [
    { label: 'Dashboard', icon: 'pi pi-fw pi-home', routerLink: '/home' },
  ];

  constructor(private authService: AuthService) {
    this.isAuthenticated = this.authService.isAuthenticated();
  }


  logOut() {
    this.authService.logout();
  }

  logIn() {
    this.authService.login();
  }
}
