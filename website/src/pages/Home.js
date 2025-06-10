// type "rcfe" and press enter to automatic create a function with the file name
import React from 'react';
import { Link } from "react-router-dom";
import "../styles/Home.css";

function Home() {
  return (
    <div className="home">
      <h1>Need to pick a movie?</h1>
      <h1>Can't decide what to watch?</h1>
      <h1>Rate some movies, and we'll give you a suggestion!</h1>
      <Link to="/survey">
        <button className="button Start"> Start </button>
      </Link>
    </div>
  );
}

export default Home;
