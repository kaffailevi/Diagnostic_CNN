import { Routes } from '@angular/router';
import { HomeComponent} from '../component/home/home.component';
import {ProfileComponent} from '../component/profile/profile.component';
import {authGuard} from '../guard/auth.guard';

export const routes: Routes = [
  {path: 'home',component:  HomeComponent},
  {path: 'profile',component:  ProfileComponent, canActivate: [authGuard]},
  {path: '', redirectTo: 'home', pathMatch: 'full'},
];
