import { Injectable } from '@angular/core';
import {BASE_URL} from '../util/urls';
import {Observable} from 'rxjs';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class AuthService {

  constructor(private http: HttpClient) { }

  login(): void {
    // Redirect the browser to the backend /login endpoint
    window.location.href = `${BASE_URL}/login`;
  }

  isAuthenticated(): Promise<boolean> {
  //     check by requesting the /my-images protected endpoint
  //     if the request is successful, the user is authenticated
    return this.http.get(`${BASE_URL}/my-images`, {withCredentials: true}).toPromise()
      .then(() => true)
      .catch(() => false);
  }


}
