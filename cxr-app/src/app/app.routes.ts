import { Routes } from '@angular/router';
import { HomeComponent} from '../component/home/home.component';
import {ProfileComponent} from '../component/profile/profile.component';

export const routes: Routes = [
  {path: 'home',component:  HomeComponent},
  {path: 'profile',component:  ProfileComponent},
  {path: '', redirectTo: 'home', pathMatch: 'full'},
];
