"use client";

import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiClient } from "@/lib/api-client";
import { Team } from "@nba/api/types";
import { Button } from "@nba/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nba/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@nba/ui/select";
import { Label } from "@nba/ui/label";
import { useToast } from "@nba/ui/use-toast";

interface GamePredictorProps {
    initialHomeTeamId?: string;
    initialAwayTeamId?: string;
}

export function GamePredictor({ initialHomeTeamId, initialAwayTeamId }: GamePredictorProps) {
  const { toast } = useToast();
  const [homeTeamId, setHomeTeamId] = useState<string>(initialHomeTeamId || "");
  const [awayTeamId, setAwayTeamId] = useState<string>(initialAwayTeamId || "");

  useEffect(() => {
    setHomeTeamId(initialHomeTeamId || "");
    setAwayTeamId(initialAwayTeamId || "");
  }, [initialHomeTeamId, initialAwayTeamId]);

  const { data: teamsData } = useQuery({
    queryKey: ["teams"],
    queryFn: () => apiClient.getTeams(30),
  });

  const mutation = useMutation({
    mutationFn: (variables: { homeTeamId: number; awayTeamId: number }) =>
      apiClient.predictGame(variables.homeTeamId, variables.awayTeamId),
    onSuccess: (data) => {
      const homeTeam = teamsData?.items.find(
        (t) => t.team_id === data.home_team_id,
      );
      const awayTeam = teamsData?.items.find(
        (t) => t.team_id === data.away_team_id,
      );
      toast({
        title: "Prediction Generated",
        description: (
          <div className="text-sm">
            <p className="font-semibold">
              {homeTeam?.team_abbrev} vs {awayTeam?.team_abbrev}
            </p>
            <p>
              <span className="font-medium">{homeTeam?.team_abbrev} Win Probability:</span>{" "}
              <span className="font-bold">{(data.home_win_probability * 100).toFixed(1)}%</span>
            </p>
            <p>
              <span className="font-medium">Predicted Score:</span>{" "}
              <span className="font-bold">{data.predicted_home_score.toFixed(0)} - {data.predicted_away_score.toFixed(0)}</span>
            </p>
          </div>
        ),
      });
    },
    onError: () => {
      toast({
        variant: "destructive",
        title: "Prediction Failed",
        description: "Could not generate prediction. Please try again.",
      });
    },
  });

  const handlePredict = () => {
    if (!homeTeamId || !awayTeamId) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please select both a home and an away team.",
      });
      return;
    }
    if (homeTeamId === awayTeamId) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Home and away teams cannot be the same.",
      });
      return;
    }
    mutation.mutate({
      homeTeamId: parseInt(homeTeamId),
      awayTeamId: parseInt(awayTeamId),
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Game Predictor</CardTitle>
        <CardDescription>
          Select two teams to predict the outcome.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="home-team">Home Team</Label>
          <Select onValueChange={setHomeTeamId} value={homeTeamId} disabled={!!initialHomeTeamId}>
            <SelectTrigger id="home-team">
              <SelectValue placeholder="Select a team" />
            </SelectTrigger>
            <SelectContent>
              {teamsData?.items.map((team: Team) => (
                <SelectItem key={team.team_id} value={String(team.team_id)}>
                  {team.full_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2">
          <Label htmlFor="away-team">Away Team</Label>
          <Select onValueChange={setAwayTeamId} value={awayTeamId} disabled={!!initialAwayTeamId}>
            <SelectTrigger id="away-team">
              <SelectValue placeholder="Select a team" />
            </SelectTrigger>
            <SelectContent>
              {teamsData?.items.map((team: Team) => (
                <SelectItem key={team.team_id} value={String(team.team_id)}>
                  {team.full_name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button
          onClick={handlePredict}
          disabled={mutation.isPending}
          className="w-full"
        >
          {mutation.isPending ? "Predicting..." : "Predict Game"}
        </Button>
      </CardContent>
    </Card>
  );
}